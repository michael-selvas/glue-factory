"""
Simply load images from a folder or nested folders (does not have any split),
and apply homographic adaptations to it. Yields an image pair without border
artifacts.
"""

import argparse
import logging
import shutil
import tarfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..geometry.homography import (
    compute_homography,
    sample_homography_corners,
    warp_points,
)
from ..models.cache_loader import CacheLoader, pad_local_features
from ..settings import DATA_PATH
from ..utils.image import read_image, read_exr, ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .augmentations import IdentityAugmentation, augmentations
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def sample_homography(img, conf: dict, size: list):
    data = {}
    H, _, coords, _ = sample_homography_corners(img.shape[:2][::-1], **conf)
    data["image"] = cv2.warpPerspective(img, H, tuple(size))
    data["H_"] = H.astype(np.float32)
    data["coords"] = coords.astype(np.float32)
    data["image_size"] = np.array(size, dtype=np.float32)
    return data


class RenderDataset(BaseDataset):
    default_conf = {
        # image search
        "data_dir": "render",  # the top-level directory
        "texture_img_dir": "tex/",
        "render_img_dir": "img/",  # the subdirectory with the images
        "render_image_list": None,  # optional: list or filename of list
        "uv_dir": "uv/",
        "check_file_exists": False,  # check if the image exists
        "glob": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        # splits
        "train_size": 100,
        "val_size": 10,
        "shuffle_seed": 0,  # or None to skip
        # image loading
        "grayscale": False,
        "right_only": False,  # image0 is orig (rescaled), image1 is right
        "reseed": False,
        "preprocessing": ImagePreprocessor.default_conf,
        "homography": {
            "difficulty": 0.8,
            "translation": 1.0,
            "max_angle": 60,
            "n_angles": 10,
            "patch_shape": [640, 480],
            "min_convexity": 0.05,
        },
        "photometric": {
            "name": "dark",
            "p": 0.75,
            # 'difficulty': 1.0,  # currently unused
        },
        # feature loading
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
            "thresh": 0.0,
            "max_num_keypoints": -1,
            "force_num_keypoints": False,
        },
    }

    def _init(self, conf):
        data_dir = DATA_PATH / conf.data_dir
        if not data_dir.exists():
            if conf.data_dir == "render":
                logger.info("Downloading the render dataset.")
                self.download_render()
            else:
                raise FileNotFoundError(data_dir)

        render_image_dir = data_dir / conf.render_img_dir
        src_image_dir = data_dir / conf.texture_img_dir
        uv_image_dir = data_dir / conf.uv_dir
        render_images = []        
        src_images = []

        if conf.render_image_list is None:
            glob = [conf.glob] if isinstance(conf.glob, str) else conf.glob
            for g in glob:
                render_images += list(render_image_dir.glob("**/" + g))

            if len(render_images) == 0:
                raise ValueError(f"Cannot find any image in folder: {render_image_dir}.")
            render_images = [i.relative_to(render_image_dir).as_posix() for i in render_images]
            render_images = sorted(render_images)  # for deterministic behavior
            logger.info("Found rendered %d images in folder.", len(render_images))
        elif isinstance(conf.render_image_list, (str, Path)):
            image_list = data_dir / conf.render_image_list
            if not image_list.exists():
                raise FileNotFoundError(f"Cannot find image list {image_list}.")
            render_images = image_list.read_text().rstrip("\n").split("\n")
            for image in render_images:
                if self.conf.check_file_exists and not (render_image_dir / image).exists():
                    raise FileNotFoundError(render_image_dir / image)
            logger.info("Found rendered %d images in list file.", len(render_images))
        elif isinstance(conf.render_image_list, omegaconf.listconfig.ListConfig):
            render_images = conf.render_image_list.to_container()
            for image in render_images:
                if self.conf.check_file_exists and not (render_image_dir / image).exists():
                    raise FileNotFoundError(render_image_dir / image)
        else:
            raise ValueError(conf.render_image_list)
        

        # print(f"src_image_dir: {src_image_dir}")
        src_images, render_images, uv_images = self._filter_src_images(render_images, src_image_dir, uv_image_dir, conf)

        render_train_images = render_images[: conf.train_size]
        render_val_images = render_images[-min(len(render_images) - conf.train_size, conf.val_size) : ]
        src_train_images = src_images[: conf.train_size]
        src_val_images = src_images[-min(len(src_images) - conf.train_size, conf.val_size) : ]
        uv_train_images = uv_images[: conf.train_size]
        uv_val_images = uv_images[-min(len(uv_images) - conf.train_size, conf.val_size) : ]
        assert len(render_train_images) == len(src_train_images) == len(uv_train_images)
        assert len(render_val_images) == len(src_val_images) == len(uv_val_images)
        print(f"render_train_images: {len(render_train_images)}, render_val_images: {len(render_val_images)}")
        train_list = list(zip(render_train_images, src_train_images, uv_train_images))
        val_list = list(zip(render_val_images, src_val_images, uv_val_images))

        if conf.shuffle_seed is not None:
            np.random.RandomState(conf.shuffle_seed).shuffle(train_list)
            np.random.RandomState(conf.shuffle_seed).shuffle(val_list)
            
        self.images = {"train": train_list, "val": val_list}

    def download_render(self):
        raise NotImplementedError(
            "The render dataset is not available for download."
        )
    
    def get_dataset(self, split):
        return _Dataset(self.conf, self.images[split], split)
    

    def _filter_src_images(self, render_images, src_image_dir, uv_image_dir, conf):
        src_images = []
        filtered_render_images = []
        uv_images = []
        for img in render_images:
            src_stem = ''
            splited_str = str(Path(img).stem).split('-')
            if 4 == len(splited_str):
                src_stem = splited_str[2]
            elif 4 < len(splited_str):
                src_stem = '-'.join(splited_str[2:-1])
            else:
                raise ValueError(f"Unexpected rendered image name: {img}")
            
            if (src_image_dir / f"{src_stem}.jpg").exists() and (uv_image_dir / f"{Path(img).stem}.exr").exists():
                src_images.append(f"{src_stem}.jpg")
                filtered_render_images.append(img)
                uv_images.append(f"{Path(img).stem}.exr")
            elif (src_image_dir / f"{src_stem}.png").exists() and (uv_image_dir / f"{Path(img).stem}.exr").exists():
                src_images.append(f"{src_stem}.png")
                filtered_render_images.append(img)
                uv_images.append(f"{Path(img).stem}.exr")
        return src_images, filtered_render_images, uv_images



class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, image_names, split):
        self.conf = conf
        self.split = split
        np_image_names = np.array(image_names)
        print(f"np_image_names: {np_image_names.shape}")
        self.src_image_names = np_image_names[:,1]
        self.render_image_names = np_image_names[:,0]
        self.uv_image_names = np_image_names[:,2]
        self.render_image_dir = DATA_PATH / conf.data_dir / conf.render_img_dir
        self.src_image_dir = DATA_PATH / conf.data_dir / conf.texture_img_dir
        self.uv_image_dir = DATA_PATH / conf.data_dir / conf.uv_dir

        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        aug_conf = conf.photometric
        aug_name = aug_conf.name
        assert (
            aug_name in augmentations.keys()
        ), f'{aug_name} not in {" ".join(augmentations.keys())}'
        self.photo_augment = augmentations[aug_name](aug_conf)
        self.left_augment = (
            IdentityAugmentation() if conf.right_only else self.photo_augment
        )
        self.img_to_tensor = IdentityAugmentation()

        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

    def _transform_keypoints(self, features, data):
        """Transform keypoints by a homography, threshold them,
        and potentially keep only the best ones."""
        # Warp points
        features["keypoints"] = warp_points(
            features["keypoints"], data["H_"], inverse=False
        )
        h, w = data["image"].shape[1:3]
        valid = (
            (features["keypoints"][:, 0] >= 0)
            & (features["keypoints"][:, 0] <= w - 1)
            & (features["keypoints"][:, 1] >= 0)
            & (features["keypoints"][:, 1] <= h - 1)
        )
        features["keypoints"] = features["keypoints"][valid]

        # Threshold
        if self.conf.load_features.thresh > 0:
            valid = features["keypoint_scores"] >= self.conf.load_features.thresh
            features = {k: v[valid] for k, v in features.items()}

        # Get the top keypoints and pad
        n = self.conf.load_features.max_num_keypoints
        if n > -1:
            inds = np.argsort(-features["keypoint_scores"])
            features = {k: v[inds[:n]] for k, v in features.items()}

            if self.conf.load_features.force_num_keypoints:
                features = pad_local_features(
                    features, self.conf.load_features.max_num_keypoints
                )

        return features

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def _read_view(self, img, H_conf, ps, left=False):

        data = sample_homography(img, H_conf, ps)
        if left:
            data["image"] = self.left_augment(data["image"], return_tensor=True)
        else:
            data["image"] = self.photo_augment(data["image"], return_tensor=True)

        gs = data["image"].new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        if self.conf.grayscale:
            data["image"] = (data["image"] * gs).sum(0, keepdim=True)

        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            features = self._transform_keypoints(features, data)
            data["cache"] = features

        return data

    def getitem(self, idx):
        src_name = self.src_image_names[idx]
        render_name = self.render_image_names[idx]
        uv_name = self.uv_image_names[idx]
        src_img = load_image(self.src_image_dir / src_name, False)
        render_img = load_image(self.render_image_dir / render_name, False)
        uv_img = read_exr(self.uv_image_dir / uv_name)
        if src_img is None:
            logging.warning("Image %s could not be read.", src_name)
            src_img = np.zeros((1024, 1024) + (() if self.conf.grayscale else (3,)))
        if render_img is None:
            logging.warning("Image %s could not be read.", render_name)
            render_img = np.zeros((1024, 1024) + (() if self.conf.grayscale else (3,)))
        if uv_name is None:
            logging.warning("Image %s could not be read.", uv_name)
            uv_img = np.zeros((1024, 1024, 3))

        src_data = self.preprocessor(src_img)
        render_data = self.preprocessor(render_img)

        # print(f"------------------------------------------------------")

        # print(f"src_data = {src_data.keys()}")
        # print(f"render_data = {render_data.keys()}")

        # print(f"src_data image size = {src_data['image_size']}")
        # print(f"render_data image size = {render_data['image_size']}")

        # print(f"src_data['original_image_size'] = {src_data['original_image_size']}")
        # print(f"render_data['original_image_size'] = {render_data['original_image_size']}")

        # print(f"src_data['scales'] = {src_data['scales']}")
        # print(f"render_data['scales'] = {render_data['scales']}")

        # print(f"src_data = {src_data['original_image_size'][0] * src_data['scales'][0]}, {src_data['original_image_size'][1] * src_data['scales'][1]}")
        # print(f"render_data = {render_data['original_image_size'][0] * render_data['scales'][0]}, {render_data['original_image_size'][1] * render_data['scales'][1]}")

        # print(f"src_img shape: {src_data['image'].shape}, render_img shape: {render_data['image'].shape}, uv_img shape: {uv_img.shape}")

        ps = self.conf.homography.patch_shape

        left_conf = omegaconf.OmegaConf.to_container(self.conf.homography)
        if self.conf.right_only:
            left_conf["difficulty"] = 0.0

        # data0 = self._read_view(src_data['image'], left_conf, ps, left=True)
        # data1 = self._read_view(src_img, self.conf.homography, ps, left=False)
        # data1 = self._read_view(render_data['image'], left_conf, ps, left=True)

        # H = compute_homography(data0["coords"], data1["coords"], [1, 1])

        data = {
            "name": src_name,
            "src_original_image_size": src_data['original_image_size'],
            "src_scales": src_data['scales'],
            "src_image_size": src_data['image_size'],
            "H_0to1": torch.eye(3),
            "idx": idx,
            "view0": { "image": src_data['image'] },
            "view1": { "image": render_data['image'] },
            "uv_data": uv_img,
            # "original_tex": src_img,
            "render_original_image_size": render_data['original_image_size'],
            "render_scales": render_data['scales'],
            "render_image_size": render_data['image_size'],
        }

        return data

    def __len__(self):
        return len(self.src_image_names)


def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = RenderDataset(conf)
    loader = dataset.get_data_loader("train")
    logger.info("The dataset has %d elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images = []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)]
            )
    plot_image_grid(images, dpi=args.dpi)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)

