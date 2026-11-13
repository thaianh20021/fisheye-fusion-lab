"""Compile and verify the actual CUDA rasterizer with gradients."""
import torch
import sys
from gsplat import rasterization

device = "cuda"
means = torch.tensor([[0., 0., 3.]], device=device, requires_grad=True)
image, alpha, _ = rasterization(
    means=means, quats=torch.tensor([[1., 0., 0., 0.]], device=device),
    scales=torch.full((1, 3), .1, device=device),
    opacities=torch.ones(1, device=device),
    colors=torch.tensor([[1., .2, .1]], device=device),
    viewmats=torch.eye(4, device=device)[None],
    Ks=torch.tensor([[[80., 0., 32.], [0., 80., 32.], [0., 0., 1.]]], device=device),
    width=64, height=64, packed=False,
    camera_model="fisheye" if "--fisheye" in sys.argv else "pinhole",
    **({"with_geer": True, "with_eval3d": True} if "--geer" in sys.argv else {}),
)
image.sum().backward()
assert image.max().item() > .1 and alpha.max().item() > .1
assert means.grad is not None and torch.isfinite(means.grad).all()
print("CUDA rasterizer and backward PASS", sys.argv[1:], torch.cuda.get_device_name(0), flush=True)
