import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")  # M1/M2/M3 GPU
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Example: simple tensor operation
x = torch.randn(3, 3).to(device)
y = torch.randn(3, 3).to(device)
z = x + y
print(z)
