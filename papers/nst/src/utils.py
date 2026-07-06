import torch
from torchvision import transforms
from PIL import Image

# Standard ImageNet Mormalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_img(img_path: str, max_size: int = 400, device: torch.device = "cuda") -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    size = max_size if max(img.size) > max_size else max(img.size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor.to(device)


def convert_to_img(tensor: torch.Tensor) -> Image.Image:
    img = tensor.cpu().clone().detach()
    img = img.squeeze(0)

    # Re-arrange back to standard numpy image layout: (C, H, W) -> (H, W, C)
    image = image.numpy().transpose(1, 2, 0)

    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = image.clip(0, 1)

    # Convert a [0, 1] float matrix into a standard [0, 255] unsigned integer image format
    return Image.fromarray((image * 255).astype('uint8'))
