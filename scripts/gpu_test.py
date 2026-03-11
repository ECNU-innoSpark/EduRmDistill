import torch

if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices available.")

if hasattr(torch, "xpu") and torch.xpu.is_available():
    print("XPU device count:", torch.xpu.device_count())
    for i in range(torch.xpu.device_count()):
        print(f"XPU device {i}: {torch.xpu.get_device_name(i)}")

if torch.backends.mps.is_available():
    print("MPS (Apple Silicon) is available.")

if hasattr(torch.backends, "rocm") and torch.backends.rocm.is_available():  # pyright: ignore[reportAttributeAccessIssue]
    print("ROCm (AMD GPU) is available.")
