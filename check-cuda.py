import torch

tensor = torch.tensor([1.0, 2.0, 3.0])

print(torch.cuda.is_available())

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")