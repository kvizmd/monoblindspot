import math

import torch


class RANSAC(torch.nn.Module):
    """
    RANSAC by Falcon Dai, we reimplemented in PyTorch.
    It is based on https://github.com/falcondai/py-ransac

    In this implementation, it supports batch-style.
    """

    def __init__(
            self,
            sampling_num: int = 3,
            max_iterations: int = 100,
            goal_inliers_ratio: float = 0.3,
            inlier_plane_distance: int = 0.05,
            prior_d: float = None):
        super().__init__()

        assert sampling_num >= 3

        self.sampling_num = sampling_num
        self.max_iterations = max_iterations
        self.goal_inliers_ratio = goal_inliers_ratio
        self.inlier_plane_distance = inlier_plane_distance
        self.prior_d = prior_d

        self.signif_logit = math.log(1 - goal_inliers_ratio)

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """
        Launch RANSAC optimization.
        data: The tensor of (B, N, 3) shape.
        """
        B, N = data.shape[:2]

        best_ic = torch.zeros(
            (B, ), dtype=torch.int64, device=data.device)
        best_model = torch.zeros(
            (B, 4), dtype=data.dtype, device=data.device)

        for i in range(self.max_iterations):
            idx = torch.randint(
                N, size=(B, self.sampling_num, 1),
                device=data.device).expand(-1, -1, 3)
            s = data.gather(1, idx)  # take the minmal sample set

            m = estimate_plane(s, self.prior_d)  # estimate model with this set

            # make inlier mask for this model
            inlier_mask = self.classify_inlier(m, data)
            ic = inlier_mask.sum(dim=1)

            update_mask = ic > best_ic
            best_ic[update_mask] = ic[update_mask]
            best_model[update_mask] = m[update_mask]

            inlier_prob = best_ic / N
            reason_itrs = self.estimate_reasonable_iteration(inlier_prob)
            if i + 1 >= reason_itrs:
                break

        return best_model

    def estimate_reasonable_iteration(self, inlier_prob: torch.Tensor) -> int:
        prob = 1 - inlier_prob.pow(self.sampling_num)
        denominator = torch.log(prob.clamp(max=0.99))
        k = self.signif_logit / (denominator + 1e-10)
        return min(int(k.max()), self.max_iterations)

    def classify_inlier(
            self,
            coeffs: torch.Tensor,
            xyz: torch.Tensor) -> torch.Tensor:
        return is_inlier(coeffs, xyz, self.inlier_plane_distance)


def augment(xyzs: torch.Tensor) -> torch.Tensor:
    axyz = torch.ones(
            list(xyzs.shape[:-1]) + [4],
            dtype=xyzs.dtype,
            device=xyzs.device)
    axyz[..., :3] = xyzs
    return axyz


def plane_distance(
        coeffs: torch.Tensor,
        xyz: torch.Tensor) -> torch.Tensor:
    shape = xyz.shape

    # To support vector and matrix, compute batch_size with reshaping.
    _coeffs = coeffs.view(-1, 1, 4)
    batch_size = _coeffs.shape[0]

    axyz = augment(xyz).view(batch_size, -1, 4).permute(0, 2, 1)

    # [B x 1 x 4] x [B x 4 x [N]] -> [B x 1 x [N]]
    return torch.bmm(_coeffs, axyz).view(*shape[:-1]).abs()


def is_inlier(
        coeffs: torch.Tensor,
        xyz: torch.Tensor,
        threshold: torch.Tensor) -> torch.Tensor:
    distance = plane_distance(coeffs, xyz)
    return distance < threshold


def estimate_plane(
        xyzs: torch.Tensor,
        prior_d: float = None) -> torch.Tensor:
    points = xyzs[..., :3]

    # homogenerous points to estimate d in ax + by + cz + d = 0.
    points = augment(points)

    U, S, Vt = torch.linalg.svd(points)
    abcd = Vt[:, -1]

    # normalize normal vector
    abcd /= abcd[:, :3].pow(2).sum(-1, True).sqrt()

    if prior_d is not None:
        abcd[:, 3] = prior_d

    # Adjust the normal vector so that it turn toward positive direction.
    neg_mask = abcd[:, 1:2] < 0
    abcd[:, :3] = (~neg_mask) * abcd[:, :3] - neg_mask * abcd[:, :3]

    return abcd


def do_faster_ransac(
        data,
        sampling_num: int = 3,
        max_iterations: int = 1000,
        plane_distance_thr: torch.Tensor = 0.05,
        prior_d: float = None):
    """
    Faster batch-style RANSAC.

    data: The tensor of (B, N, 3) shape.
    """
    assert sampling_num >= 3

    B, N, D = data.shape
    indices = torch.randint(
        N, size=(B, max_iterations * sampling_num),
        device=data.device)

    # take the minmal sample set
    s = data.gather(1, indices[..., None].expand(-1, -1, D))

    # Pack iterations into the batches
    s = s.view(B * max_iterations, sampling_num, 3)

    model = estimate_plane(s, prior_d)  # estimate model with this set

    # make inlier mask for this model
    data = data.unsqueeze(1).expand(-1, max_iterations, -1, -1)
    data = data.reshape(-1, N, 3)

    if isinstance(plane_distance_thr, torch.Tensor):
        plane_distance_thr = \
            plane_distance_thr.view(-1, 1).expand(-1, max_iterations)
        plane_distance_thr = plane_distance_thr.reshape(-1, 1)

    inlier_mask = is_inlier(model, data, plane_distance_thr)
    scores = inlier_mask.sum(dim=1)

    # Unpack batches
    model = model.view(B, max_iterations, -1)
    scores = scores.view(B, max_iterations)

    best_idx = scores.argmax(dim=1)
    best_model = model.gather(
        1, best_idx.view(B, 1, 1).expand(-1, -1, model.shape[-1]))[:, 0]
    return best_model
