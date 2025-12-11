import numpy as np
import torch
import os
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .depth import project, sample_depth
from .epipolar import T_to_E, sym_epipolar_distance_all
from .homography import warp_points_torch

IGNORE_FEATURE = -2
UNMATCHED_FEATURE = -1


@torch.no_grad()
def find_pixel_mapping_list(uv_data, original_image_size, kp, tolerance=0.01, inverse=False):
    before = torch.cuda.memory_allocated() / 1024**2  # MB 단위
    # print(f"Before: {before:.2f} MB")
    device = uv_data.device
    B = kp.shape[0]
    kp = kp.to(device)
 
    if isinstance(original_image_size, torch.Tensor) and original_image_size.ndim == 2:
        W_list, H_list = original_image_size[:, 0].tolist(), original_image_size[:, 1].tolist()
    else:
        W, H = original_image_size
        H_list, W_list = [H]*B, [W]*B

    batched_matches = torch.full((B, kp.shape[1], 2), -1, dtype=torch.int32, device=device)

    if inverse:
        for b in range(B):
            H, W = H_list[b], W_list[b]
            kp_b = kp[b].long()
            uv_b = uv_data[b]

            assert uv_b.shape[-1] >= 3

            kp_b[:, 0] = kp_b[:, 0].clamp(0, uv_b.shape[1] - 1)
            kp_b[:, 1] = kp_b[:, 1].clamp(0, uv_b.shape[0] - 1)

            uvs = uv_b[kp_b[:, 1], kp_b[:, 0]]   # (N, 3)
            mask = (uvs == 0).all(dim=1)
            x = (uvs[:, 0] * W).round().clamp(0, W - 1)
            y = ((1.0 - uvs[:, 1]) * H).round().clamp(0, H - 1)

            batched_matches[b] = torch.stack([x, y], dim=-1).int()
            batched_matches[b][mask] = torch.tensor((W-1, H-1), device=device, dtype=torch.int32)

        after = torch.cuda.memory_allocated() / 1024**2
        # print(f"After: {after:.2f} MB")
        # print(f"Memory used: {after - before:.2f} MB")
        return batched_matches.to(device)
    # 후보 픽셀 (uv_data[...,2] == 1)만 모으기

    for b in range(B):
        H, W = H_list[b], W_list[b]
        uv_b = uv_data[b]
        kp_b = kp[b]

        assert uv_b.shape[-1] >= 3

        mask = uv_b[..., 2] == 1
        if not mask.any():
            continue

        # 후보 (u, v)
        uv_candidates = uv_b[..., :2][mask]  # (Nc, 2)
        coords = torch.nonzero(mask, as_tuple=False)  # (Nc, 2) — (r, c)

        # keypoints → [0,1]로 변환
        u = (kp_b[:, 0] / W)
        v = (1.0 - kp_b[:, 1] / H)
        query = torch.stack([u, v], dim=-1)  # (N, 2)

        query_cpu = query.cpu().numpy()  # GPU → CPU
        uv_candidates_cpu = uv_candidates.cpu().numpy()
        tree = cKDTree(uv_candidates_cpu)
        distances, indices = tree.query(query_cpu, k=1)  # 각 query에 대해 가장 가까운 1개 찾기
        min_dists = torch.from_numpy(distances).to(device)  # 다시 GPU로
        indices = torch.from_numpy(indices).to(device)

        # 거리 계산 (broadcasted)
        # dists = torch.cdist(query, uv_candidates)  # (N, Nc)
        # min_dists, indices = dists.min(dim=1)

        # # tolerance 이내인 점만 유효
        valid = min_dists <= tolerance
        matched_coords = torch.full((len(query), 2), -1, dtype=torch.int32, device=device)
        matched_coords[valid] = coords[indices[valid]].int()

        # # (r, c) → (x, y)
        matched_xy = matched_coords[:, [1, 0]]
        batched_matches[b] = matched_xy

    after = torch.cuda.memory_allocated() / 1024**2
    # print(f"NO::After: {after:.2f} MB")
    # print(f"NO::Memory used: {after - before:.2f} MB")
    return batched_matches    


def debug_save_matching_image(view0, view1, kp0, kp1, m0, m1, save_dir="debug_matching"):
    import cv2
    os.makedirs(save_dir, exist_ok=True)
    B = view0.shape[0]

    # 텐서를 numpy로 변환 (채널 순서와 스케일 조정)
    def to_bgr(img_tensor):
        img = img_tensor.detach().cpu()
        if img.ndim == 4:  # (B, C, H, W)
            img = img.permute(0, 2, 3, 1)
        img = (img * 255.0).clamp(0, 255).byte().numpy()
        if img.shape[-1] == 1:
            img = [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in img]
        return img
    

    imgs0 = to_bgr(view0)
    imgs1 = to_bgr(view1)

    for b in range(B):
        img0 = imgs0[b].copy()
        img1 = imgs1[b].copy()

        kp0_b = kp0[b].detach().cpu().numpy()
        kp1_b = kp1[b].detach().cpu().numpy()
        N = min(len(kp0_b), len(kp1_b))

        # kp0 (view0) 그리기 - 파란 점
        colors = (np.random.rand(N, 3) * 255).astype(np.uint8)

        for idx in range(N):
            k1_idx = m0[b, idx].item()
            if k1_idx < 0 or k1_idx >= N:
                continue
            x0, y0 = map(int, kp0_b[idx])
            x1, y1 = map(int, kp1_b[k1_idx])

            color = tuple(int(c) for c in colors[idx])
            # view0 / view1 각각에 같은 색으로 표시
            cv2.circle(img0, (x0, y0), 3, color, -1)
            cv2.circle(img1, (x1, y1), 3, color, -1)

        # 저장
        cv2.imwrite(os.path.join(save_dir, f"view0_{b:02d}_kp0.png"), img0)
        cv2.imwrite(os.path.join(save_dir, f"view1_{b:02d}_kp1.png"), img1)

        print(f"[DEBUG] Saved keypoint visualization for batch {b} → {save_dir}")

def debug_save_line_matching_image(view0, view1, kp0, kp1, m0, m1, save_dir="debug_matching"):
    import cv2
    os.makedirs(save_dir, exist_ok=True)
    B = view0.shape[0]
    print(f"matching shape: {m0.shape}, {m1.shape}")
    print(f"kp shape: {kp0.shape}, {kp1.shape}")

    # 텐서를 numpy로 변환 (채널 순서와 스케일 조정)
    def to_bgr(img_tensor):
        img = img_tensor.detach().cpu()
        if img.ndim == 4:  # (B, C, H, W)
            img = img.permute(0, 2, 3, 1)
        img = (img * 255.0).clamp(0, 255).byte().numpy()
        if img.shape[-1] == 1:
            img = [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in img]
        return img
    

    imgs0 = to_bgr(view0)
    imgs1 = to_bgr(view1)

    for b in range(B):
        img0 = imgs0[b].copy()
        img1 = imgs1[b].copy()

        kp0_b = kp0[b].detach().cpu().numpy()
        kp1_b = kp1[b].detach().cpu().numpy()
        N = min(len(kp0_b), len(kp1_b))

        # kp0 (view0) 그리기 - 파란 점
        colors = (np.random.rand(N, 3) * 255).astype(np.uint8)

        for idx in range(N):
            k1_idx = m0[b, idx].item()
            if k1_idx < 0 or k1_idx >= N:
                continue
            x0_0, y0_0, x0_1, y0_1 = map(int, kp0_b[idx])
            x1_0, y1_0, x1_1, y1_1 = map(int, kp1_b[k1_idx])

            color = tuple(int(c) for c in colors[idx])
            # view0 / view1 각각에 같은 색으로 표시
            cv2.circle(img0, (x0_0, y0_0), 3, color, -1)
            cv2.circle(img0, (x0_1, y0_1), 3, color, -1)
            cv2.circle(img1, (x1_0, y1_0), 3, color, -1)
            cv2.circle(img1, (x1_1, y1_1), 3, color, -1)
            cv2.line(img0, (x0_0, y0_0), (x0_1, y0_1), color, 1)
            cv2.line(img1, (x1_0, y1_0), (x1_1, y1_1), color, 1)

        # 저장
        cv2.imwrite(os.path.join(save_dir, f"view0_{b:02d}_kp0.png"), img0)
        cv2.imwrite(os.path.join(save_dir, f"view1_{b:02d}_kp1.png"), img1)

        print(f"[DEBUG] Saved keypoint visualization for batch {b} → {save_dir}")


def debug_save_kp_image(view0, view1, kp0, kp0_1, save_dir="debug_kp"):
    import cv2
    os.makedirs(save_dir, exist_ok=True)
    B = view0.shape[0]


    # 텐서를 numpy로 변환 (채널 순서와 스케일 조정)
    def to_bgr(img_tensor):
        img = img_tensor.detach().cpu()
        if img.ndim == 4:  # (B, C, H, W)
            img = img.permute(0, 2, 3, 1)
        img = (img * 255.0).clamp(0, 255).byte().numpy()
        if img.shape[-1] == 1:
            img = [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in img]
        return img

    imgs0 = to_bgr(view0)
    imgs1 = to_bgr(view1)

    for b in range(B):
        img0 = imgs0[b].copy()
        img1 = imgs1[b].copy()

        kp0_b = kp0[b].detach().cpu().numpy()
        kp0_1_b = kp0_1[b].detach().cpu().numpy()
        N = min(len(kp0_b), len(kp0_1_b))

        # kp0 (view0) 그리기 - 파란 점
        colors = (np.random.rand(N, 3) * 255).astype(np.uint8)

        for i in range(N):
            x0, y0 = map(int, kp0_b[i])
            x1, y1 = map(int, kp0_1_b[i])
            color = tuple(int(c) for c in colors[i])

            # view0 / view1 각각에 같은 색으로 표시
            cv2.circle(img0, (x0, y0), 3, color, -1)
            cv2.circle(img1, (x1, y1), 3, color, -1)

        # 저장
        cv2.imwrite(os.path.join(save_dir, f"view0_{b:02d}_kp0.png"), img0)
        cv2.imwrite(os.path.join(save_dir, f"view1_{b:02d}_kp0_1.png"), img1)

        print(f"[DEBUG] Saved keypoint visualization for batch {b} → {save_dir}")

@torch.no_grad()
def gt_matches_from_3d_render(
    kp0, kp1, uv_data, view0, view1, original_image_size, src_image_size, src_scales, pos_th=3, neg_th=5, tolerance=0.01, epi_th=None, cc_th=None, **kw
):
    """
    uv_data와 original_tex를 사용하여 두 이미지 간의 ground truth 특징점 매칭을 생성합니다.
    
    Args:
        kp0, kp1: 첫 번째/두 번째 이미지의 키포인트 좌표 [B, N, 2]
        data: uv_data와 original_tex를 포함한 데이터 딕셔너리
        pos_th: positive 매칭으로 간주할 픽셀 거리 임계값 (기본값: 3픽셀)
        neg_th: negative 매칭으로 간주할 픽셀 거리 임계값 (기본값: 5픽셀)
        tolerance: UV 좌표 매칭 허용 오차 (기본값: 0.01)
        epi_th: epipolar geometry를 사용한 추가 필터링 임계값
        cc_th: cross-check 임계값
    
    Returns:
        dict: assignment matrix, matches, scores 등을 포함한 결과
    """

    # 키포인트가 없는 경우 빈 결과 반환
    if kp0.shape[1] == 0 or kp1.shape[1] == 0:
        b_size, n_kp0 = kp0.shape[:2]
        n_kp1 = kp1.shape[1]
        # 빈 assignment matrix 생성
        assignment = torch.zeros(
            b_size, n_kp0, n_kp1, dtype=torch.bool, device=kp0.device
        )
        # 모든 키포인트를 unmatched로 설정 (-1)
        m0 = -torch.ones_like(kp0[:, :, 0]).long()
        m1 = -torch.ones_like(kp1[:, :, 0]).long()
        return assignment, m0, m1
        
    kp0_orig = kp0 / src_scales[:, None, :]

    kp0_1 = find_pixel_mapping_list(uv_data, original_image_size, kp0_orig)
    # kp1을 view0으로 투영 (역방향 호모그래피)  
    kp1_orig = find_pixel_mapping_list(uv_data, original_image_size, kp1, inverse=True)   # [B, N, 2]
    kp1_0 = kp1_orig * src_scales[:, None, :]

    # debug_save_kp_image(view0, view1, kp0, kp0_1, "debug_kp0")
    # debug_save_kp_image(view0, view1, kp1_0, kp1, "debug_kp1")


    
    # 거리 행렬 생성 [B, M, N] - 모든 키포인트 쌍 간의 거리 계산
    # kp0를 투영한 점과 실제 kp1 간의 유클리드 거리의 제곱
    # print(f"kp0_1 shape: {kp0_1.shape}, kp1 shape: {kp1.shape}")
    dist0 = torch.sum((kp0_1.unsqueeze(-2) - kp1.unsqueeze(-3)) ** 2, -1)
    # kp1을 투영한 점과 실제 kp0 간의 유클리드 거리의 제곱
    dist1 = torch.sum((kp0.unsqueeze(-2) - kp1_0.unsqueeze(-3)) ** 2, -1)
    # 양방향 거리 중 큰 값을 사용 (대칭적 일관성 보장)
    dist = torch.max(dist0, dist1)

    # 매칭 품질 점수 계산 (positive: +1, negative: -1, 중간: 0)
    reward = (dist < pos_th**2).float() - (dist > neg_th**2).float()

    # 각 키포인트의 최근접 매칭 찾기
    # 각 kp0에 대해 가장 가까운 kp1의 인덱스
    min0 = dist.min(-1).indices  # [B, M]
    # 각 kp1에 대해 가장 가까운 kp0의 인덱스
    min1 = dist.min(-2).indices  # [B, N]

    # 상호 최근접 매칭(mutual nearest neighbor) 확인을 위한 마스크 생성
    ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
    ismin1 = ismin0.clone()
    # kp0 -> kp1 방향의 최근접 매칭 표시
    ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
    # kp1 -> kp0 방향의 최근접 매칭 표시  
    ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
    # positive 매칭: 상호 최근접이면서 거리가 pos_th 이하인 경우
    positive = ismin0 & ismin1 & (dist < pos_th**2)

    # negative 매칭 판별: 가장 가까운 점도 neg_th보다 멀리 있는 경우
    negative0 = dist0.min(-1).values > neg_th**2  # [B, M]
    negative1 = dist1.min(-2).values > neg_th**2  # [B, N]

    # 매칭 결과를 인덱스로 패키징
    # 매칭 상태 정의: -1(unmatched), -2(ignore) 
    unmatched = min0.new_tensor(UNMATCHED_FEATURE)  # -1
    ignore = min0.new_tensor(IGNORE_FEATURE)        # -2
    
    # 각 키포인트의 최종 매칭 상태 결정
    # positive 매칭이 있으면 해당 인덱스, 없으면 ignore로 초기화
    m0 = torch.where(positive.any(-1), min0, ignore)  # [B, M]
    m1 = torch.where(positive.any(-2), min1, ignore)  # [B, N]
    
    # negative로 판별된 점들은 명시적으로 unmatched로 설정
    m0 = torch.where(negative0, unmatched, m0)
    m1 = torch.where(negative1, unmatched, m1)

    # debug_save_matching_image(view0, view1, kp0, kp1, m0, m1, "debug_matching")

    return {
        "assignment": positive,           # [B, M, N] positive 매칭의 boolean 행렬
        "reward": reward,                 # [B, M, N] 매칭 품질 점수 (-1, 0, +1)
        "matches0": m0,                   # [B, M] view0 키포인트의 매칭 인덱스 (-1: unmatched, -2: ignore)
        "matches1": m1,                   # [B, N] view1 키포인트의 매칭 인덱스  
        "matching_scores0": (m0 > -1).float(),  # [B, M] view0의 매칭 신뢰도 (0 또는 1)
        "matching_scores1": (m1 > -1).float(),  # [B, N] view1의 매칭 신뢰도
        "proj_0to1": kp0_1,              # [B, M, 2] view0 키포인트를 view1으로 투영한 좌표
        "proj_1to0": kp1_0,              # [B, N, 2] view1 키포인트를 view0으로 투영한 좌표
    }




@torch.no_grad()
def gt_line_matches_from_3d_render(
    pred_lines0,
    pred_lines1,
    valid_lines0,
    valid_lines1,
    uv_data, 
    original_image_size,
    src_scales,
    view0,
    view1,
    H,
    npts=50,
    dist_th=5,
    overlap_th=0.2,
    min_visibility_th=0.2,
):
    """
    3D 렌더링 데이터로부터 ground truth 선 매칭을 계산합니다.

    이 함수는 두 이미지의 예측된 선들을 입력으로 받아, 제공된 호모그래피 행렬을 사용하여
    한 이미지의 선들을 다른 이미지로 투영합니다. 재투영 후 근접성과 중첩 기준에 따라 선들을 매칭합니다.
    나머지 선들은 가시성과 유효성에 따라 UNMATCHED 또는 IGNORE로 라벨링됩니다.

    Args:
        pred_lines0 (torch.Tensor): 첫 번째 이미지의 예측된 선들, 형태 [B, N, 4] 또는 [B, N, 2, 2].
        pred_lines1 (torch.Tensor): 두 번째 이미지의 예측된 선들, 형태 [B, N, 4] 또는 [B, N, 2, 2].
        valid_lines0 (torch.Tensor): 첫 번째 이미지에서 유효한 선들을 나타내는 불리언 마스크, 형태 [B, N].
        valid_lines1 (torch.Tensor): 두 번째 이미지에서 유효한 선들을 나타내는 불리언 마스크, 형태 [B, N].
        uv_data: 사용되지 않는 매개변수 (UV 좌표 데이터, 이 함수에서 활용되지 않음).
        original_image_size: 사용되지 않는 매개변수 (원본 이미지 크기).
        src_scales: 사용되지 않는 매개변수 (소스 스케일).
        shape0 (tuple): 첫 번째 이미지의 형태 (높이, 너비).
        shape1 (tuple): 두 번째 이미지의 형태 (높이, 너비).
        H (torch.Tensor): 이미지 간 점 변환을 위한 호모그래피 행렬, 형태 [B, 3, 3].
        npts (int, optional): 매칭을 위한 각 선을 따라 샘플링할 점의 수. 기본값은 50.
        dist_th (float, optional): 점들이 가까운 것으로 간주하는 거리 임계값. 기본값은 5.
        overlap_th (float, optional): 선 매칭을 위한 중첩 임계값. 기본값은 0.2.
        min_visibility_th (float, optional): 선의 최소 가시성 임계값. 기본값은 0.2.

    Returns:
        tuple:
            - positive (torch.Tensor): 매칭된 선 쌍의 불리언 마스크, 형태 [B, N_lines0, N_lines1].
            - m0 (torch.Tensor): 첫 번째 이미지 선들의 매칭 인덱스, 형태 [B, N_lines0].
            - m1 (torch.Tensor): 두 번째 이미지 선들의 매칭 인덱스, 형태 [B, N_lines1].

    Notes:
        - UNMATCHED: 재투영이 이미지 외부에 있거나 다른 선과 멀리 떨어진 선들.
        - IGNORE: 유효하지 않은 것으로 라벨링된 선들.
    """
    # 이미지 크기 추출
    h0, w0 = view0.shape[-2:]
    h1, w1 = view1.shape[-2:]
    # 선 데이터를 복사하여 수정 가능하게 함
    lines0 = pred_lines0.clone()
    lines1 = pred_lines1.clone()
    # 선 형태를 표준화 ([x1,y1,x2,y2] 형태로)
    if lines0.shape[-2:] == (2, 2):
        lines0 = torch.flatten(lines0, -2)
    elif lines0.dim() == 4:
        lines0 = torch.cat([lines0[:, :, 0], lines0[:, :, -1]], dim=2)
    # 같은 방식으로 lines1 처리
    if lines1.shape[-2:] == (2, 2):
        lines1 = torch.flatten(lines1, -2)
    elif lines1.dim() == 4:
        lines1 = torch.cat([lines1[:, :, 0], lines1[:, :, -1]], dim=2)
    # 배치 크기와 선 개수 추출
    b_size, n_lines0, _ = lines0.shape
    b_size, n_lines1, _ = lines1.shape

    # 선 좌표를 이미지 경계 내로 클리핑
    lines0 = torch.min(
        torch.max(lines0, torch.zeros_like(lines0)),
        lines0.new_tensor([w0 - 1, h0 - 1, w0 - 1, h0 - 1], dtype=torch.float),
    )
    lines1 = torch.min(
        torch.max(lines1, torch.zeros_like(lines1)),
        lines1.new_tensor([w1 - 1, h1 - 1, w1 - 1, h1 - 1], dtype=torch.float),
    )

    # 각 선을 따라 점 샘플링
    pts0 = sample_pts(lines0, npts).reshape(b_size, n_lines0 * npts, 2)
    pts1 = sample_pts(lines1, npts).reshape(b_size, n_lines1 * npts, 2)

    # 점들을 다른 이미지로 투영
    # pts0_1 = warp_points_torch(pts0, H, inverse=False)
    # pts1_0 = warp_points_torch(pts1, H, inverse=True)

    # 점들을 다른 이미지로 투영
    pts0_orig = pts0 / src_scales[:, None, :]
    pts0_1 = find_pixel_mapping_list(uv_data, original_image_size, pts0_orig)
    # kp1을 view0으로 투영 (역방향 호모그래피)  
    pts1_orig = find_pixel_mapping_list(uv_data, original_image_size, pts1, inverse=True)   # [B, N, 2]
    pts1_0 = pts1_orig * src_scales[:, None, :]

    # debug_save_kp_image(view0, view1, pts0, pts0_1, "debug_pts0")
    # debug_save_kp_image(view0, view1, pts1_0, pts1, "debug_pts1")


    pts0_1 = pts0_1.reshape(b_size, n_lines0, npts, 2)
    pts1_0 = pts1_0.reshape(b_size, n_lines1, npts, 2)

    # 선의 가시성이 min_visibility_th보다 낮으면 OUTSIDE로 간주
    pts_out_of0 = (pts1_0 < 0).any(-1) | (
        pts1_0 >= torch.tensor([w0, h0]).to(pts1_0)
    ).any(-1)
    pts_out_of0 = pts_out_of0.reshape(b_size, n_lines1, npts).float()
    out_of0 = pts_out_of0.mean(dim=-1) >= (1 - min_visibility_th)
    pts_out_of1 = (pts0_1 < 0).any(-1) | (
        pts0_1 >= torch.tensor([w1, h1]).to(pts0_1)
    ).any(-1)
    pts_out_of1 = pts_out_of1.reshape(b_size, n_lines0, npts).float()
    out_of1 = pts_out_of1.mean(dim=-1) >= (1 - min_visibility_th)

    # 재투영된 점들이 원래 선에 가까운지 계산
    perp_dists0, overlaping0 = torch_perp_dist(lines0, pts1_0)
    close_points0 = (perp_dists0 < dist_th) & overlaping0  # [bs, nl0, nl1, npts]
    del perp_dists0, overlaping0

    perp_dists1, overlaping1 = torch_perp_dist(lines1, pts0_1)
    close_points1 = (perp_dists1 < dist_th) & overlaping1  # [bs, nl1, nl0, npts]
    del perp_dists1, overlaping1
    torch.cuda.empty_cache()

    # 각 선에 대해 재투영된 점들 중 가까운 점의 개수 계산
    num_close_pts0 = close_points0.sum(dim=-1)  # [bs, nl0, nl1]
    # num_close_pts0_t = num_close_pts0.transpose(-1, -2)
    # 각 선에 대해 재투영된 점들 중 가까운 점의 개수 계산 (역방향)
    num_close_pts1 = close_points1.sum(dim=-1)
    num_close_pts1_t = num_close_pts1.transpose(-1, -2)  # [bs, nl1, nl0]

    num_close_pts = num_close_pts0 * num_close_pts1_t
    mask_close = (
        (num_close_pts1_t > npts * overlap_th)
        & (num_close_pts0 > npts * overlap_th)
        & ~out_of0.unsqueeze(1)
        & ~out_of1.unsqueeze(-1)
    )

    # 매칭되지 않은 선 정의
    unmatched0 = torch.all(~mask_close, dim=2) | out_of1
    unmatched1 = torch.all(~mask_close, dim=1) | out_of0

    # 무시할 선 정의
    ignore0 = ~valid_lines0
    ignore1 = ~valid_lines1

    cost = -num_close_pts.clone()
    # 매칭되지 않거나 유효하지 않은 선에 높은 비용 부여
    cost[unmatched0] = 1e6
    cost[ignore0] = 1e6
    cost = cost.transpose(1, 2)
    cost[unmatched1] = 1e6
    cost[ignore1] = 1e6
    cost = cost.transpose(1, 2)
    # 각 행에 대해 최대 점 개수의 열 반환 (선형 할당)
    assignation = np.array(
        [linear_sum_assignment(C) for C in cost.detach().cpu().numpy()]
    )
    assignation = torch.tensor(assignation).to(num_close_pts)

    # 매칭되지 않은 라벨 설정
    unmatched = assignation.new_tensor(UNMATCHED_FEATURE)
    ignore = assignation.new_tensor(IGNORE_FEATURE)

    positive = num_close_pts.new_zeros(num_close_pts.shape, dtype=torch.bool)
    # TODO Do with a single and beautiful call
    # for b in range(b_size):
    #     positive[b][assignation[b, 0], assignation[b, 1]] = True
    positive[
        torch.arange(b_size)[:, None].repeat(1, assignation.shape[-1]).flatten(),
        assignation[:, 0].flatten(),
        assignation[:, 1].flatten(),
    ] = True

    # 매칭 인덱스 생성
    m0 = assignation.new_full((b_size, n_lines0), unmatched, dtype=torch.long)
    m0.scatter_(-1, assignation[:, 0], assignation[:, 1])
    m1 = assignation.new_full((b_size, n_lines1), unmatched, dtype=torch.long)
    m1.scatter_(-1, assignation[:, 1], assignation[:, 0])

    positive = positive & mask_close
    # 무시하거나 매칭되지 않은 값 제거
    positive[unmatched0] = False
    positive[ignore0] = False
    positive = positive.transpose(1, 2)
    positive[unmatched1] = False
    positive[ignore1] = False
    positive = positive.transpose(1, 2)
    m0[~positive.any(-1)] = unmatched
    m0[unmatched0] = unmatched
    m0[ignore0] = ignore
    m1[~positive.any(-2)] = unmatched
    m1[unmatched1] = unmatched
    m1[ignore1] = ignore

    # debug_save_line_matching_image(view0, view1, lines0, lines1, m0, m1, "debug_line_matching")

    if num_close_pts.numel() == 0:
        no_matches = torch.zeros(positive.shape[0], 0).to(positive)
        return positive, no_matches, no_matches

    return positive, m0, m1


@torch.no_grad()
def gt_matches_from_pose_depth(
    kp0, kp1, data, pos_th=3, neg_th=5, epi_th=None, cc_th=None, **kw
):
    """
    카메라 포즈와 깊이 정보를 사용하여 두 이미지 간의 ground truth 특징점 매칭을 생성합니다.
    
    Args:
        kp0, kp1: 첫 번째/두 번째 이미지의 키포인트 좌표 [B, N, 2]
        data: 카메라 정보, 포즈, 깊이 맵을 포함한 데이터 딕셔너리
        pos_th: positive 매칭으로 간주할 픽셀 거리 임계값 (기본값: 3픽셀)
        neg_th: negative 매칭으로 간주할 픽셀 거리 임계값 (기본값: 5픽셀)
        epi_th: epipolar geometry를 사용한 추가 필터링 임계값
        cc_th: cross-check 임계값
    
    Returns:
        dict: assignment matrix, matches, scores 등을 포함한 결과
    """
    
    # 키포인트가 없는 경우 빈 결과 반환
    if kp0.shape[1] == 0 or kp1.shape[1] == 0:
        b_size, n_kp0 = kp0.shape[:2]
        n_kp1 = kp1.shape[1]
        # 빈 assignment matrix 생성
        assignment = torch.zeros(
            b_size, n_kp0, n_kp1, dtype=torch.bool, device=kp0.device
        )
        # 모든 키포인트를 unmatched로 설정 (-1)
        m0 = -torch.ones_like(kp0[:, :, 0]).long()
        m1 = -torch.ones_like(kp1[:, :, 0]).long()
        return assignment, m0, m1
        
    # 카메라 파라미터와 포즈 변환 행렬 추출
    camera0, camera1 = data["view0"]["camera"], data["view1"]["camera"]
    T_0to1, T_1to0 = data["T_0to1"], data.get("T_1to0", data["T_0to1"].inv())

    # 깊이 맵 정보 가져오기
    depth0 = data["view0"].get("depth")
    depth1 = data["view1"].get("depth")
    
    # 미리 계산된 키포인트별 깊이값이 있으면 사용, 없으면 샘플링
    if "depth_keypoints0" in kw and "depth_keypoints1" in kw:
        d0, valid0 = kw["depth_keypoints0"], kw["valid_depth_keypoints0"]
        d1, valid1 = kw["depth_keypoints1"], kw["valid_depth_keypoints1"]
    else:
        assert depth0 is not None
        assert depth1 is not None
        # 키포인트 위치에서 깊이값 샘플링 (bilinear interpolation)
        d0, valid0 = sample_depth(kp0, depth0)
        d1, valid1 = sample_depth(kp1, depth1)

    # 3D 좌표를 상대방 이미지 평면으로 투영
    # kp0의 3D 점들을 camera1으로 투영 -> kp0_1
    kp0_1, visible0 = project(
        kp0, d0, depth1, camera0, camera1, T_0to1, valid0, ccth=cc_th
    )
    # kp1의 3D 점들을 camera0으로 투영 -> kp1_0  
    kp1_0, visible1 = project(
        kp1, d1, depth0, camera1, camera0, T_1to0, valid1, ccth=cc_th
    )
    # 양방향 모두 보이는 점들만 고려
    mask_visible = visible0.unsqueeze(-1) & visible1.unsqueeze(-2)

    # 거리 행렬 생성 [B, M, N] - M개의 kp0와 N개의 kp1 간의 모든 조합
    # kp0를 투영한 점과 실제 kp1 간의 거리
    dist0 = torch.sum((kp0_1.unsqueeze(-2) - kp1.unsqueeze(-3)) ** 2, -1)
    # kp1을 투영한 점과 실제 kp0 간의 거리  
    dist1 = torch.sum((kp0.unsqueeze(-2) - kp1_0.unsqueeze(-3)) ** 2, -1)
    # 양방향 거리 중 큰 값을 사용 (더 엄격한 조건)
    dist = torch.max(dist0, dist1)
    inf = dist.new_tensor(float("inf"))
    # 보이지 않는 점들은 무한대 거리로 설정
    dist = torch.where(mask_visible, dist, inf)

    # 각 키포인트의 최근접 매칭 찾기
    # 각 kp0에 대해 가장 가까운 kp1의 인덱스
    min0 = dist.min(-1).indices  # [B, M]
    # 각 kp1에 대해 가장 가까운 kp0의 인덱스  
    min1 = dist.min(-2).indices  # [B, N]

    # 상호 최근접 매칭 확인을 위한 마스크 생성
    ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
    ismin1 = ismin0.clone()
    # kp0 -> kp1 방향의 최근접 매칭 표시
    ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
    # kp1 -> kp0 방향의 최근접 매칭 표시
    ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
    # positive 매칭: 상호 최근접이면서 거리가 임계값 이하
    positive = ismin0 & ismin1 & (dist < pos_th**2)

    # negative 매칭 판별: 가장 가까운 점도 임계값보다 멀리 있는 경우
    negative0 = (dist0.min(-1).values > neg_th**2) & valid0  # [B, M]
    negative1 = (dist1.min(-2).values > neg_th**2) & valid1  # [B, N]

    # 매칭 결과를 인덱스로 패키징
    # 매칭 상태 정의: -1(unmatched), -2(ignore)
    unmatched = min0.new_tensor(UNMATCHED_FEATURE)  # -1
    ignore = min0.new_tensor(IGNORE_FEATURE)        # -2
    
    # 각 키포인트의 매칭 상태 결정
    # positive 매칭이 있으면 해당 인덱스, 없으면 ignore로 초기화
    m0 = torch.where(positive.any(-1), min0, ignore)  # [B, M]
    m1 = torch.where(positive.any(-2), min1, ignore)  # [B, N]
    
    # negative로 판별된 점들은 명시적으로 unmatched로 설정
    m0 = torch.where(negative0, unmatched, m0)
    m1 = torch.where(negative1, unmatched, m1)

    # Fundamental matrix 계산 (epipolar geometry)
    F = (
        camera1.calibration_matrix().inverse().transpose(-1, -2)
        @ T_to_E(T_0to1)  # Essential matrix로 변환
        @ camera0.calibration_matrix().inverse()
    )
    # 모든 키포인트 쌍에 대한 대칭적 에피폴라 거리 계산
    epi_dist = sym_epipolar_distance_all(kp0, kp1, F)

    # 에피폴라 기하학을 사용한 추가적인 unmatched 점 찾기
    if epi_th is not None:
        # ignore 상태인 점들에 대해서만 에피폴라 거리 고려
        mask_ignore = (m0.unsqueeze(-1) == ignore) & (m1.unsqueeze(-2) == ignore)
        epi_dist = torch.where(mask_ignore, epi_dist, inf)
        
        # 에피폴라 거리가 임계값보다 큰 점들 찾기
        exclude0 = epi_dist.min(-1).values > neg_th
        exclude1 = epi_dist.min(-2).values > neg_th
        
        # 유효하지 않으면서 에피폴라 제약도 만족하지 않는 점들을 unmatched로 설정
        m0 = torch.where((~valid0) & exclude0, ignore.new_tensor(-1), m0)
        m1 = torch.where((~valid1) & exclude1, ignore.new_tensor(-1), m1)

    return {
        "assignment": positive,           # [B, M, N] positive 매칭의 boolean 행렬
        "reward": (dist < pos_th**2).float() - (epi_dist > neg_th).float(),  # 매칭 품질 점수
        "matches0": m0,                   # [B, M] view0 키포인트의 매칭 인덱스 (-1: unmatched, -2: ignore)
        "matches1": m1,                   # [B, N] view1 키포인트의 매칭 인덱스
        "matching_scores0": (m0 > -1).float(),  # [B, M] view0의 매칭 신뢰도 (0 또는 1)
        "matching_scores1": (m1 > -1).float(),  # [B, N] view1의 매칭 신뢰도
        "depth_keypoints0": d0,           # [B, M] view0 키포인트의 깊이값
        "depth_keypoints1": d1,           # [B, N] view1 키포인트의 깊이값  
        "proj_0to1": kp0_1,              # [B, M, 2] view0 키포인트를 view1으로 투영한 좌표
        "proj_1to0": kp1_0,              # [B, N, 2] view1 키포인트를 view0으로 투영한 좌표
        "visible0": visible0,             # [B, M] view0 키포인트가 view1에서 보이는지 여부
        "visible1": visible1,             # [B, N] view1 키포인트가 view0에서 보이는지 여부
    }


@torch.no_grad()
def gt_matches_from_homography(kp0, kp1, H, pos_th=3, neg_th=6, **kw):
    """
    호모그래피 변환을 사용하여 두 이미지 간의 ground truth 특징점 매칭을 생성합니다.
    평면 장면이나 카메라가 회전만 하는 경우에 적합합니다.
    
    Args:
        kp0, kp1: 첫 번째/두 번째 이미지의 키포인트 좌표 [B, N, 2]
        H: view0에서 view1으로의 호모그래피 변환 행렬 [B, 3, 3] 또는 [3, 3]
        pos_th: positive 매칭으로 간주할 픽셀 거리 임계값 (기본값: 3픽셀)  
        neg_th: negative 매칭으로 간주할 픽셀 거리 임계값 (기본값: 6픽셀)
    
    Returns:
        dict: assignment matrix, matches, scores, projections 등을 포함한 결과
    """
    
    # 키포인트가 없는 경우 빈 결과 반환
    if kp0.shape[1] == 0 or kp1.shape[1] == 0:
        b_size, n_kp0 = kp0.shape[:2]
        n_kp1 = kp1.shape[1]
        # 빈 assignment matrix 생성
        assignment = torch.zeros(
            b_size, n_kp0, n_kp1, dtype=torch.bool, device=kp0.device
        )
        # 모든 키포인트를 unmatched로 설정 (-1)
        m0 = -torch.ones_like(kp0[:, :, 0]).long()
        m1 = -torch.ones_like(kp1[:, :, 0]).long()
        return assignment, m0, m1
        
    # 호모그래피를 사용하여 키포인트들을 상대방 이미지로 투영
    # kp0를 view1으로 투영 (순방향 호모그래피)
    kp0_1 = warp_points_torch(kp0, H, inverse=False)  # [B, M, 2]
    # kp1을 view0으로 투영 (역방향 호모그래피)  
    kp1_0 = warp_points_torch(kp1, H, inverse=True)   # [B, N, 2]

    # 거리 행렬 생성 [B, M, N] - 모든 키포인트 쌍 간의 거리 계산
    # kp0를 투영한 점과 실제 kp1 간의 유클리드 거리의 제곱
    dist0 = torch.sum((kp0_1.unsqueeze(-2) - kp1.unsqueeze(-3)) ** 2, -1)
    # kp1을 투영한 점과 실제 kp0 간의 유클리드 거리의 제곱
    dist1 = torch.sum((kp0.unsqueeze(-2) - kp1_0.unsqueeze(-3)) ** 2, -1)
    # 양방향 거리 중 큰 값을 사용 (대칭적 일관성 보장)
    dist = torch.max(dist0, dist1)

    # 매칭 품질 점수 계산 (positive: +1, negative: -1, 중간: 0)
    reward = (dist < pos_th**2).float() - (dist > neg_th**2).float()

    # 각 키포인트의 최근접 매칭 찾기
    # 각 kp0에 대해 가장 가까운 kp1의 인덱스
    min0 = dist.min(-1).indices  # [B, M]
    # 각 kp1에 대해 가장 가까운 kp0의 인덱스
    min1 = dist.min(-2).indices  # [B, N]

    # 상호 최근접 매칭(mutual nearest neighbor) 확인을 위한 마스크 생성
    ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
    ismin1 = ismin0.clone()
    # kp0 -> kp1 방향의 최근접 매칭 표시
    ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
    # kp1 -> kp0 방향의 최근접 매칭 표시  
    ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
    # positive 매칭: 상호 최근접이면서 거리가 pos_th 이하인 경우
    positive = ismin0 & ismin1 & (dist < pos_th**2)

    # negative 매칭 판별: 가장 가까운 점도 neg_th보다 멀리 있는 경우
    negative0 = dist0.min(-1).values > neg_th**2  # [B, M]
    negative1 = dist1.min(-2).values > neg_th**2  # [B, N]

    # 매칭 결과를 인덱스로 패키징
    # 매칭 상태 정의: -1(unmatched), -2(ignore) 
    unmatched = min0.new_tensor(UNMATCHED_FEATURE)  # -1
    ignore = min0.new_tensor(IGNORE_FEATURE)        # -2
    
    # 각 키포인트의 최종 매칭 상태 결정
    # positive 매칭이 있으면 해당 인덱스, 없으면 ignore로 초기화
    m0 = torch.where(positive.any(-1), min0, ignore)  # [B, M]
    m1 = torch.where(positive.any(-2), min1, ignore)  # [B, N]
    
    # negative로 판별된 점들은 명시적으로 unmatched로 설정
    m0 = torch.where(negative0, unmatched, m0)
    m1 = torch.where(negative1, unmatched, m1)

    return {
        "assignment": positive,           # [B, M, N] positive 매칭의 boolean 행렬
        "reward": reward,                 # [B, M, N] 매칭 품질 점수 (-1, 0, +1)
        "matches0": m0,                   # [B, M] view0 키포인트의 매칭 인덱스 (-1: unmatched, -2: ignore)
        "matches1": m1,                   # [B, N] view1 키포인트의 매칭 인덱스  
        "matching_scores0": (m0 > -1).float(),  # [B, M] view0의 매칭 신뢰도 (0 또는 1)
        "matching_scores1": (m1 > -1).float(),  # [B, N] view1의 매칭 신뢰도
        "proj_0to1": kp0_1,              # [B, M, 2] view0 키포인트를 view1으로 투영한 좌표
        "proj_1to0": kp1_0,              # [B, N, 2] view1 키포인트를 view0으로 투영한 좌표
    }


def sample_pts(lines, npts):
    dir_vec = (lines[..., 2:4] - lines[..., :2]) / (npts - 1)
    pts = lines[..., :2, np.newaxis] + dir_vec[..., np.newaxis].expand(
        dir_vec.shape + (npts,)
    ) * torch.arange(npts).to(lines)
    pts = torch.transpose(pts, -1, -2)
    return pts


def torch_perp_dist(segs2d, points_2d):
    # Check batch size and segments format
    assert segs2d.shape[0] == points_2d.shape[0]
    assert segs2d.shape[-1] == 4
    dir = segs2d[..., 2:] - segs2d[..., :2]
    sizes = torch.norm(dir, dim=-1).half()
    norm_dir = dir / torch.unsqueeze(sizes, dim=-1)
    # middle_ptn = 0.5 * (segs2d[..., 2:] + segs2d[..., :2])
    # centered [batch, nsegs0, nsegs1, n_sampled_pts, 2]
    centered = points_2d[:, None] - segs2d[..., None, None, 2:]

    R = torch.cat(
        [
            norm_dir[..., 0, None],
            norm_dir[..., 1, None],
            -norm_dir[..., 1, None],
            norm_dir[..., 0, None],
        ],
        dim=2,
    ).reshape((len(segs2d), -1, 2, 2))
    # Try to reduce the memory consumption by using float16 type
    if centered.is_cuda:
        centered, R = centered.half(), R.half()
    # R: [batch, nsegs0, 2, 2] , centered: [batch, nsegs1, n_sampled_pts, 2]
    #    -> [batch, nsegs0, nsegs1, n_sampled_pts, 2]
    rotated = torch.einsum("bdji,bdepi->bdepj", R, centered)

    overlaping = (rotated[..., 0] <= 0) & (
        torch.abs(rotated[..., 0]) <= sizes[..., None, None]
    )

    return torch.abs(rotated[..., 1]), overlaping


@torch.no_grad()
def gt_line_matches_from_pose_depth(
    pred_lines0,
    pred_lines1,
    valid_lines0,
    valid_lines1,
    data,
    npts=50,
    dist_th=5,
    overlap_th=0.2,
    min_visibility_th=0.5,
):
    """Compute ground truth line matches and label the remaining the lines as:
    - UNMATCHED: if reprojection is outside the image
                 or far away from any other line.
    - IGNORE: if a line has not enough valid depth pixels along itself
              or it is labeled as invalid."""
    lines0 = pred_lines0.clone()
    lines1 = pred_lines1.clone()

    if pred_lines0.shape[1] == 0 or pred_lines1.shape[1] == 0:
        bsize, nlines0, nlines1 = (
            pred_lines0.shape[0],
            pred_lines0.shape[1],
            pred_lines1.shape[1],
        )
        positive = torch.zeros(
            (bsize, nlines0, nlines1), dtype=torch.bool, device=pred_lines0.device
        )
        m0 = torch.full((bsize, nlines0), -1, device=pred_lines0.device)
        m1 = torch.full((bsize, nlines1), -1, device=pred_lines0.device)
        return positive, m0, m1

    if lines0.shape[-2:] == (2, 2):
        lines0 = torch.flatten(lines0, -2)
    elif lines0.dim() == 4:
        lines0 = torch.cat([lines0[:, :, 0], lines0[:, :, -1]], dim=2)
    if lines1.shape[-2:] == (2, 2):
        lines1 = torch.flatten(lines1, -2)
    elif lines1.dim() == 4:
        lines1 = torch.cat([lines1[:, :, 0], lines1[:, :, -1]], dim=2)
    b_size, n_lines0, _ = lines0.shape
    b_size, n_lines1, _ = lines1.shape
    h0, w0 = data["view0"]["depth"][0].shape
    h1, w1 = data["view1"]["depth"][0].shape

    lines0 = torch.min(
        torch.max(lines0, torch.zeros_like(lines0)),
        lines0.new_tensor([w0 - 1, h0 - 1, w0 - 1, h0 - 1], dtype=torch.float),
    )
    lines1 = torch.min(
        torch.max(lines1, torch.zeros_like(lines1)),
        lines1.new_tensor([w1 - 1, h1 - 1, w1 - 1, h1 - 1], dtype=torch.float),
    )

    # Sample points along each line
    pts0 = sample_pts(lines0, npts).reshape(b_size, n_lines0 * npts, 2)
    pts1 = sample_pts(lines1, npts).reshape(b_size, n_lines1 * npts, 2)

    # Sample depth and valid points
    d0, valid0_pts0 = sample_depth(pts0, data["view0"]["depth"])
    d1, valid1_pts1 = sample_depth(pts1, data["view1"]["depth"])

    # Reproject to the other view
    pts0_1, visible0 = project(
        pts0,
        d0,
        data["view1"]["depth"],
        data["view0"]["camera"],
        data["view1"]["camera"],
        data["T_0to1"],
        valid0_pts0,
    )
    pts1_0, visible1 = project(
        pts1,
        d1,
        data["view0"]["depth"],
        data["view1"]["camera"],
        data["view0"]["camera"],
        data.get(data["T_1to0"], data["T_0to1"].inv()),
        valid1_pts1,
    )

    h0, w0 = data["view0"]["image"].shape[-2:]
    h1, w1 = data["view1"]["image"].shape[-2:]
    # If a line has less than min_visibility_th inside the image is considered OUTSIDE
    pts_out_of0 = (pts1_0 < 0).any(-1) | (
        pts1_0 >= torch.tensor([w0, h0]).to(pts1_0)
    ).any(-1)
    pts_out_of0 = pts_out_of0.reshape(b_size, n_lines1, npts).float()
    out_of0 = pts_out_of0.mean(dim=-1) >= (1 - min_visibility_th)
    pts_out_of1 = (pts0_1 < 0).any(-1) | (
        pts0_1 >= torch.tensor([w1, h1]).to(pts0_1)
    ).any(-1)
    pts_out_of1 = pts_out_of1.reshape(b_size, n_lines0, npts).float()
    out_of1 = pts_out_of1.mean(dim=-1) >= (1 - min_visibility_th)

    # visible0 is [bs, nl0 * npts]
    pts0_1 = pts0_1.reshape(b_size, n_lines0, npts, 2)
    pts1_0 = pts1_0.reshape(b_size, n_lines1, npts, 2)

    perp_dists0, overlaping0 = torch_perp_dist(lines0, pts1_0)
    close_points0 = (perp_dists0 < dist_th) & overlaping0  # [bs, nl0, nl1, npts]
    del perp_dists0, overlaping0
    close_points0 = close_points0 * visible1.reshape(b_size, 1, n_lines1, npts)

    perp_dists1, overlaping1 = torch_perp_dist(lines1, pts0_1)
    close_points1 = (perp_dists1 < dist_th) & overlaping1  # [bs, nl1, nl0, npts]
    del perp_dists1, overlaping1
    close_points1 = close_points1 * visible0.reshape(b_size, 1, n_lines0, npts)
    torch.cuda.empty_cache()

    # For each segment detected in 0, how many sampled points from
    # reprojected segments 1 are close
    num_close_pts0 = close_points0.sum(dim=-1)  # [bs, nl0, nl1]

    # num_close_pts0_t = num_close_pts0.transpose(-1, -2)
    # For each segment detected in 1, how many sampled points from
    # reprojected segments 0 are close
    num_close_pts1 = close_points1.sum(dim=-1)
    num_close_pts1_t = num_close_pts1.transpose(-1, -2)  # [bs, nl1, nl0]
    num_close_pts = num_close_pts0 * num_close_pts1_t
    mask_close = (
        num_close_pts1_t
        > visible0.reshape(b_size, n_lines0, npts).float().sum(-1)[:, :, None]
        * overlap_th
    ) & (
        num_close_pts0
        > visible1.reshape(b_size, n_lines1, npts).float().sum(-1)[:, None] * overlap_th
    )
    # mask_close = (num_close_pts1_t > npts * overlap_th) & (
    # num_close_pts0 > npts * overlap_th)

    # Define the unmatched lines
    unmatched0 = torch.all(~mask_close, dim=2) | out_of1
    unmatched1 = torch.all(~mask_close, dim=1) | out_of0

    # Define the lines to ignore
    ignore0 = (
        valid0_pts0.reshape(b_size, n_lines0, npts).float().mean(dim=-1)
        < min_visibility_th
    ) | ~valid_lines0
    ignore1 = (
        valid1_pts1.reshape(b_size, n_lines1, npts).float().mean(dim=-1)
        < min_visibility_th
    ) | ~valid_lines1

    cost = -num_close_pts.clone()
    # High score for unmatched and non-valid lines
    cost[unmatched0] = 1e6
    cost[ignore0] = 1e6
    # TODO: Is it reasonable to forbid the matching with a segment because it
    #  has not GT depth?
    cost = cost.transpose(1, 2)
    cost[unmatched1] = 1e6
    cost[ignore1] = 1e6
    cost = cost.transpose(1, 2)

    # For each row, returns the col of max number of points
    assignation = np.array(
        [linear_sum_assignment(C) for C in cost.detach().cpu().numpy()]
    )
    assignation = torch.tensor(assignation).to(num_close_pts)
    # Set ignore and unmatched labels
    unmatched = assignation.new_tensor(UNMATCHED_FEATURE)
    ignore = assignation.new_tensor(IGNORE_FEATURE)

    positive = num_close_pts.new_zeros(num_close_pts.shape, dtype=torch.bool)
    all_in_batch = (
        torch.arange(b_size)[:, None].repeat(1, assignation.shape[-1]).flatten()
    )
    positive[all_in_batch, assignation[:, 0].flatten(), assignation[:, 1].flatten()] = (
        True
    )

    m0 = assignation.new_full((b_size, n_lines0), unmatched, dtype=torch.long)
    m0.scatter_(-1, assignation[:, 0], assignation[:, 1])
    m1 = assignation.new_full((b_size, n_lines1), unmatched, dtype=torch.long)
    m1.scatter_(-1, assignation[:, 1], assignation[:, 0])

    positive = positive & mask_close
    # Remove values to be ignored or unmatched
    positive[unmatched0] = False
    positive[ignore0] = False
    positive = positive.transpose(1, 2)
    positive[unmatched1] = False
    positive[ignore1] = False
    positive = positive.transpose(1, 2)
    m0[~positive.any(-1)] = unmatched
    m0[unmatched0] = unmatched
    m0[ignore0] = ignore
    m1[~positive.any(-2)] = unmatched
    m1[unmatched1] = unmatched
    m1[ignore1] = ignore

    if num_close_pts.numel() == 0:
        no_matches = torch.zeros(positive.shape[0], 0).to(positive)
        return positive, no_matches, no_matches

    return positive, m0, m1


@torch.no_grad()
def gt_line_matches_from_homography(
    pred_lines0,
    pred_lines1,
    valid_lines0,
    valid_lines1,
    shape0,
    shape1,
    H,
    npts=50,
    dist_th=5,
    overlap_th=0.2,
    min_visibility_th=0.2,
):
    """Compute ground truth line matches and label the remaining the lines as:
    - UNMATCHED: if reprojection is outside the image or far away from any other line.
    - IGNORE: if a line is labeled as invalid."""
    h0, w0 = shape0[-2:]
    h1, w1 = shape1[-2:]
    lines0 = pred_lines0.clone()
    lines1 = pred_lines1.clone()
    if lines0.shape[-2:] == (2, 2):
        lines0 = torch.flatten(lines0, -2)
    elif lines0.dim() == 4:
        lines0 = torch.cat([lines0[:, :, 0], lines0[:, :, -1]], dim=2)
    if lines1.shape[-2:] == (2, 2):
        lines1 = torch.flatten(lines1, -2)
    elif lines1.dim() == 4:
        lines1 = torch.cat([lines1[:, :, 0], lines1[:, :, -1]], dim=2)
    b_size, n_lines0, _ = lines0.shape
    b_size, n_lines1, _ = lines1.shape

    lines0 = torch.min(
        torch.max(lines0, torch.zeros_like(lines0)),
        lines0.new_tensor([w0 - 1, h0 - 1, w0 - 1, h0 - 1], dtype=torch.float),
    )
    lines1 = torch.min(
        torch.max(lines1, torch.zeros_like(lines1)),
        lines1.new_tensor([w1 - 1, h1 - 1, w1 - 1, h1 - 1], dtype=torch.float),
    )

    # Sample points along each line
    pts0 = sample_pts(lines0, npts).reshape(b_size, n_lines0 * npts, 2)
    pts1 = sample_pts(lines1, npts).reshape(b_size, n_lines1 * npts, 2)

    # Project the points to the other image
    pts0_1 = warp_points_torch(pts0, H, inverse=False)
    pts1_0 = warp_points_torch(pts1, H, inverse=True)
    pts0_1 = pts0_1.reshape(b_size, n_lines0, npts, 2)
    pts1_0 = pts1_0.reshape(b_size, n_lines1, npts, 2)

    # If a line has less than min_visibility_th inside the image is considered OUTSIDE
    pts_out_of0 = (pts1_0 < 0).any(-1) | (
        pts1_0 >= torch.tensor([w0, h0]).to(pts1_0)
    ).any(-1)
    pts_out_of0 = pts_out_of0.reshape(b_size, n_lines1, npts).float()
    out_of0 = pts_out_of0.mean(dim=-1) >= (1 - min_visibility_th)
    pts_out_of1 = (pts0_1 < 0).any(-1) | (
        pts0_1 >= torch.tensor([w1, h1]).to(pts0_1)
    ).any(-1)
    pts_out_of1 = pts_out_of1.reshape(b_size, n_lines0, npts).float()
    out_of1 = pts_out_of1.mean(dim=-1) >= (1 - min_visibility_th)

    perp_dists0, overlaping0 = torch_perp_dist(lines0, pts1_0)
    close_points0 = (perp_dists0 < dist_th) & overlaping0  # [bs, nl0, nl1, npts]
    del perp_dists0, overlaping0

    perp_dists1, overlaping1 = torch_perp_dist(lines1, pts0_1)
    close_points1 = (perp_dists1 < dist_th) & overlaping1  # [bs, nl1, nl0, npts]
    del perp_dists1, overlaping1
    torch.cuda.empty_cache()

    # For each segment detected in 0,
    # how many sampled points from reprojected segments 1 are close
    num_close_pts0 = close_points0.sum(dim=-1)  # [bs, nl0, nl1]
    # num_close_pts0_t = num_close_pts0.transpose(-1, -2)
    # For each segment detected in 1,
    # how many sampled points from reprojected segments 0 are close
    num_close_pts1 = close_points1.sum(dim=-1)
    num_close_pts1_t = num_close_pts1.transpose(-1, -2)  # [bs, nl1, nl0]

    num_close_pts = num_close_pts0 * num_close_pts1_t
    mask_close = (
        (num_close_pts1_t > npts * overlap_th)
        & (num_close_pts0 > npts * overlap_th)
        & ~out_of0.unsqueeze(1)
        & ~out_of1.unsqueeze(-1)
    )

    # Define the unmatched lines
    unmatched0 = torch.all(~mask_close, dim=2) | out_of1
    unmatched1 = torch.all(~mask_close, dim=1) | out_of0

    # Define the lines to ignore
    ignore0 = ~valid_lines0
    ignore1 = ~valid_lines1

    cost = -num_close_pts.clone()
    # High score for unmatched and non-valid lines
    cost[unmatched0] = 1e6
    cost[ignore0] = 1e6
    cost = cost.transpose(1, 2)
    cost[unmatched1] = 1e6
    cost[ignore1] = 1e6
    cost = cost.transpose(1, 2)
    # For each row, returns the col of max number of points
    assignation = np.array(
        [linear_sum_assignment(C) for C in cost.detach().cpu().numpy()]
    )
    assignation = torch.tensor(assignation).to(num_close_pts)

    # Set unmatched labels
    unmatched = assignation.new_tensor(UNMATCHED_FEATURE)
    ignore = assignation.new_tensor(IGNORE_FEATURE)

    positive = num_close_pts.new_zeros(num_close_pts.shape, dtype=torch.bool)
    # TODO Do with a single and beautiful call
    # for b in range(b_size):
    #     positive[b][assignation[b, 0], assignation[b, 1]] = True
    positive[
        torch.arange(b_size)[:, None].repeat(1, assignation.shape[-1]).flatten(),
        assignation[:, 0].flatten(),
        assignation[:, 1].flatten(),
    ] = True

    m0 = assignation.new_full((b_size, n_lines0), unmatched, dtype=torch.long)
    m0.scatter_(-1, assignation[:, 0], assignation[:, 1])
    m1 = assignation.new_full((b_size, n_lines1), unmatched, dtype=torch.long)
    m1.scatter_(-1, assignation[:, 1], assignation[:, 0])

    positive = positive & mask_close
    # Remove values to be ignored or unmatched
    positive[unmatched0] = False
    positive[ignore0] = False
    positive = positive.transpose(1, 2)
    positive[unmatched1] = False
    positive[ignore1] = False
    positive = positive.transpose(1, 2)
    m0[~positive.any(-1)] = unmatched
    m0[unmatched0] = unmatched
    m0[ignore0] = ignore
    m1[~positive.any(-2)] = unmatched
    m1[unmatched1] = unmatched
    m1[ignore1] = ignore

    if num_close_pts.numel() == 0:
        no_matches = torch.zeros(positive.shape[0], 0).to(positive)
        return positive, no_matches, no_matches

    return positive, m0, m1
