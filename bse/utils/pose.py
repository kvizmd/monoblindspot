import torch


def transformation_from_parameters(
        axisangle: torch.Tensor,
        translation: torch.Tensor,
        invert: bool = False) -> torch.Tensor:
    """
    Convert axisangle and translation into a rigid matrix.
    """

    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(vec: torch.Tensor) -> torch.Tensor:
    """
    Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(
        (vec.shape[0], 4, 4), dtype=vec.dtype, device=vec.device)
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = vec.contiguous().view(-1, 3, 1)
    return T


def rot_from_axisangle(vec: torch.Tensor) -> torch.Tensor:
    """
    Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """

    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros(
        (vec.shape[0], 4, 4), dtype=vec.dtype, device=vec.device)
    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def scaling_from_parameters(scale: torch.Tensor) -> torch.Tensor:
    T = torch.zeros(
        (scale.shape[0], 4, 4), dtype=scale.dtype, device=scale.device)
    T[:, 0, 0] = scale[..., 0].view(-1)
    T[:, 1, 1] = scale[..., 1].view(-1)
    T[:, 2, 2] = scale[..., 2].view(-1)
    T[:, 3, 3] = 1
    return T
