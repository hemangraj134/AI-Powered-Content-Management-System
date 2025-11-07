import torch
import sys

print(f"--- GPU Verification ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"Torch Version: {torch.__version__}")

is_available = torch.cuda.is_available()
print(f"Is CUDA (GPU) Available? {is_available}")

if is_available:
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("--- WARNING: Torch cannot find your NVIDIA GPU. ---")
    print("--- (This is a problem we need to fix!) ---")
print("------------------------")
