import collections.abc as collections
from pathlib import Path
from typing import Optional, Tuple

import os
import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ..geometry.wrappers import Camera
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    print("Warning: OpenEXR not found. EXR 파일을 직접 읽을 수 없습니다.")
    print("대신 OpenCV로 EXR 파일 읽기를 시도합니다.")
    HAS_OPENEXR = False

class ImagePreprocessor:
    default_conf = {
        "resize": None,  # target edge length, None for no resizing
        "edge_divisible_by": None,
        "side": "long",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
        "square_pad": False,
        "add_padding_mask": False,
    }

    def __init__(self, conf) -> None:
        super().__init__()
        default_conf = OmegaConf.create(self.default_conf)
        OmegaConf.set_struct(default_conf, True)
        self.conf = OmegaConf.merge(default_conf, conf)

    def __call__(self, img: torch.Tensor, interpolation: Optional[str] = None) -> dict:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        size = h, w
        if self.conf.resize is not None:
            if interpolation is None:
                interpolation = self.conf.interpolation
            size = self.get_new_image_size(h, w)
            img = kornia.geometry.transform.resize(
                img,
                size,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
                interpolation=interpolation,
            )
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        T = np.diag([scale[0], scale[1], 1])

        data = {
            "scales": scale,
            "image_size": np.array(size[::-1]),
            "transform": T,
            "original_image_size": np.array([w, h]),
        }
        if self.conf.square_pad:
            sl = max(img.shape[-2:])
            data["image"] = torch.zeros(
                *img.shape[:-2], sl, sl, device=img.device, dtype=img.dtype
            )
            data["image"][:, : img.shape[-2], : img.shape[-1]] = img
            if self.conf.add_padding_mask:
                data["padding_mask"] = torch.zeros(
                    *img.shape[:-3], 1, sl, sl, device=img.device, dtype=torch.bool
                )
                data["padding_mask"][:, : img.shape[-2], : img.shape[-1]] = True

        else:
            data["image"] = img
        return data

    def load_image(self, image_path: Path) -> dict:
        return self(load_image(image_path))

    def get_new_image_size(
        self,
        h: int,
        w: int,
    ) -> Tuple[int, int]:
        side = self.conf.side
        if isinstance(self.conf.resize, collections.Iterable):
            assert len(self.conf.resize) == 2
            return tuple(self.conf.resize)
        side_size = self.conf.resize
        aspect_ratio = w / h
        if side not in ("short", "long", "vert", "horz"):
            raise ValueError(
                f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'"
            )
        if side == "vert":
            size = side_size, int(side_size * aspect_ratio)
        elif side == "horz":
            size = int(side_size / aspect_ratio), side_size
        elif (side == "short") ^ (aspect_ratio < 1.0):
            size = side_size, int(side_size * aspect_ratio)
        else:
            size = int(side_size / aspect_ratio), side_size

        if self.conf.edge_divisible_by is not None:
            df = self.conf.edge_divisible_by
            size = list(map(lambda x: int(x // df * df), size))
        return size


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def read_exr_opencv(exr_path):
    """
    OpenCV를 사용하여 EXR 파일 읽기 (OpenEXR이 없을 때)
    """
    try:
        # OpenCV로 EXR 읽기 (32비트 부동소수점)
        img = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"파일을 읽을 수 없습니다: {exr_path}")
        
        # BGR to RGB 변환 (OpenCV는 BGR 순서)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    except Exception as e:
        print(f"OpenCV로 EXR 읽기 실패: {e}")
        return None

def read_exr_openexr(exr_path):
    """
    OpenEXR 라이브러리를 사용하여 EXR 파일 읽기
    """
    try:
        exr_file = OpenEXR.InputFile(exr_path)
        header = exr_file.header()
        
        # 이미지 크기 가져오기
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
        # 채널 정보 가져오기
        channels = list(header['channels'].keys())
        # print(f"EXR 채널: {channels}")
        
        # RGB 채널 읽기
        if 'R' in channels and 'G' in channels and 'B' in channels:
            r_str = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
            g_str = exr_file.channel('G', Imath.PixelType(Imath.PixelType.FLOAT))  
            b_str = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
            
            # 바이트 문자열을 numpy 배열로 변환
            r = np.frombuffer(r_str, dtype=np.float32).reshape(size[1], size[0])
            g = np.frombuffer(g_str, dtype=np.float32).reshape(size[1], size[0])
            b = np.frombuffer(b_str, dtype=np.float32).reshape(size[1], size[0])
            
            # RGB 이미지로 결합
            img = np.stack([r, g, b], axis=-1)
            return img
        else:
            print(f"RGB 채널을 찾을 수 없습니다. 사용 가능한 채널: {channels}")
            return None
            
    except Exception as e:
        print(f"OpenEXR로 EXR 읽기 실패: {e}")
        return None

def read_exr(path: Path):
    """
    EXR 파일 읽기 (사용 가능한 라이브러리 자동 선택)
    """
    exr_path = str(path)
    if not os.path.exists(exr_path):
        print(f"파일이 존재하지 않습니다: {exr_path}")
        return None
        
    # print(f"EXR 파일 읽기: {exr_path}")
    
    # OpenEXR이 있으면 우선 사용
    if HAS_OPENEXR:
        result = read_exr_openexr(exr_path)
        if result is not None:
            return result
    
    # OpenEXR이 실패하거나 없으면 OpenCV 사용
    return read_exr_opencv(exr_path)

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def load_image(path: Path, grayscale=False) -> torch.Tensor:
    image = read_image(path, grayscale=grayscale)
    return numpy_image_to_torch(image)


def grid_sample(
    image: torch.Tensor,
    coords: torch.Tensor,
    interpolation: str = "bilinear",
    align_corners: bool = False,
):
    assert image.dim() == coords.dim()
    is_batched = image.dim() == 4

    if is_batched:
        assert coords.dim() == 4
        return F.grid_sample(
            image.to(coords.device), coords, interpolation, align_corners=False
        )
    else:
        return F.grid_sample(
            image[None].to(coords.device),
            coords[None],
            mode=interpolation,
            align_corners=align_corners,
        )[0]


def get_pixel_grid(
    *,
    fmap: torch.Tensor | None = None,  # B x H X W X D
    camera: Camera | None = None,
    size: Optional[tuple[int, int]] = None,
    device: torch.device | None = None,
    dtype=torch.float32,
    normalized: bool = False,
) -> torch.Tensor:
    if fmap is None:
        if camera is None:
            if size is None:
                raise ValueError("Specify fmap, size, or camera")
            w, h = size
        else:
            w, h = camera.size.int()
            device = camera.device
            dtype = camera.dtype
    else:
        *_, h, w, _ = fmap.shape
        device = fmap.device
        dtype = fmap.dtype
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(w, dtype=dtype, device=device),
            torch.arange(h, dtype=dtype, device=device),
            indexing="xy",
        ),
        dim=-1,
    )
    if fmap.ndim == 4:
        b = fmap.shape[0]
        grid = grid[None].expand(b, -1, -1, -1)
    grid += 0.5
    if normalized:
        grid *= 2 / grid.new_tensor([w, h])
        grid -= 1
    elif fmap is not None and camera is not None:
        # In case fmaps are at a different image resolution
        grid *= camera.size / grid.new_tensor([w, h])
    return grid


def chw_from_hwc(coords):
    # ...HWC -> ...CHW
    return coords.transpose(-2, -1).transpose(-3, -2)


def hwc_from_chw(image):
    # ...CHW -> ...HWC
    return image.transpose(-3, -2).transpose(-2, -1)


def denormalize_coords(coords, hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Denormalize coordinates from [-1, 1] to [0, H] or [0, W] (COLMAP)"""
    coords = coords.clone()
    if hw is None:
        hw = coords.shape[-3:-1]
    coords[..., 0] = (coords[..., 0] + 1) / 2 * (hw[1] - 1)
    coords[..., 1] = (coords[..., 1] + 1) / 2 * (hw[0] - 1)
    return coords


def normalize_coords(coords, hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Normalize coordinates from [0, H] or [0, W] (COLMAP) to [-1, 1]"""
    coords = coords.clone()
    if hw is None:
        hw = coords.shape[-3:-1]
    coords[..., 0] = coords[..., 0] / (hw[1] - 1) * 2 - 1
    coords[..., 1] = coords[..., 1] / (hw[0] - 1) * 2 - 1
    return coords


def cycle_dist(
    q_to_ref: torch.Tensor, ref_to_q: torch.Tensor, normalized: bool = False
) -> torch.Tensor:
    """Compute cycle consistency error between two coordinate fields."""
    q_to_ref_to_q = hwc_from_chw(grid_sample(chw_from_hwc(ref_to_q), q_to_ref))

    return torch.linalg.norm(
        get_pixel_grid(fmap=q_to_ref, normalized=normalized)
        - (q_to_ref_to_q if normalized else denormalize_coords(q_to_ref_to_q)),
        dim=-1,
    )
