import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

# This is your original manual transform, renamed and preserved as an nn.Module.
# It's designed to work on batches of tensors, making it suitable for
# exporting or including directly in a model's forward pass.
class ManualTransform(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # x: (N, 3, H, W), float32, range [0,1]

        n, c, h, w = x.shape

        # --- Resize: shorter side -> 280 (aspect preserved)
        if h < w:
            new_h = 280
            new_w = int(w * 280 / h)
        else:
            new_w = 280
            new_h = int(h * 280 / w)

        x = F.interpolate(
            x,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True  # Use antialias for better quality
        )

        # --- CenterCrop (256, 256)
        # We round to match torchvision's behavior, ensuring correct centering.
        top = int(round((new_h - 256) / 2.0))
        left = int(round((new_w - 256) / 2.0))
        x = x[:, :, top:top+256, left:left+256]

        # --- Normalize
        x = (x - self.mean) / self.std

        return x

# This is the new wrapper class that makes ManualTransform compatible with
# torchvision's data loaders. It handles the PIL -> Tensor conversion and
# batching/unbatching steps automatically.
class DataLoaderCompatibleTransform:
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()
        self.tensor_transform = ManualTransform()
        # Ensure the module is in evaluation mode and not tracking gradients
        self.tensor_transform.eval()

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        # 1. Convert PIL image to a tensor
        img_tensor = self.to_tensor(pil_img)

        # 2. Add a batch dimension (N, C, H, W)
        img_tensor_batch = img_tensor.unsqueeze(0)

        # 3. Apply the tensor-based transform
        with torch.no_grad():
            transformed_tensor_batch = self.tensor_transform(img_tensor_batch)

        # 4. Remove the batch dimension
        transformed_tensor = transformed_tensor_batch.squeeze(0)

        return transformed_tensor


if __name__ == "__main__":
    # Here is how you can use your new transform with ImageFolder:
    
    print("Initializing DataLoaderCompatibleTransform...")
    manual_transform = DataLoaderCompatibleTransform()
    
    print("Applying the transform to the dataset...")
    # Now you can pass your manual_transform directly to ImageFolder
    weather_images = torchvision.datasets.ImageFolder(
        root='dataset', 
        transform=manual_transform
    )
    
    print(f"Successfully loaded {len(weather_images)} images with the manual transform.")
    
    # You can now access an item to see the result:
    first_image_tensor, first_label = weather_images[0]
    
    print("\nExample transformed tensor:")
    print("Shape:", first_image_tensor.shape)
    print("Data (top-left 5x5 of first channel):")
    print(first_image_tensor[0, :5, :5])

    # For comparison, here is the original torchvision transform
    from transforms import test_transform
    weather_images_torch = torchvision.datasets.ImageFolder(
        root='dataset',
        transform=test_transform
    )
    torch_image_tensor, _ = weather_images_torch[0]
    print("\nFor comparison, the original torchvision transform output:")
    print(torch_image_tensor[0, :5, :5])

    # With the fixes to rounding and antialiasing, the difference should be minimal
    assert torch.allclose(first_image_tensor, torch_image_tensor, atol=1e-4), \
        "Tensors are still not close enough. Further investigation needed."
    
    print("\nAssertion passed: The output of your manual transform is consistent with the torchvision transform.")