import torch

print("CUDA available:", torch.cuda.is_available())
print("XPU available:", torch.xpu.is_available() if hasattr(torch, "xpu") else False)
print("MPS available:", torch.backends.mps.is_available())
print(
    "Rocm available:",
    torch.backends.rocm.is_available() if hasattr(torch.backends, "rocm") else False,  # type: ignore[reportAttributeError]
)
