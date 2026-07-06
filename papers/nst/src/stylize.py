import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os

from .utils import load_img, convert_to_img


class ImageFeature(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = vgg[:1]    # conv1_1
        self.slice2 = vgg[1:6]   # conv2_1
        self.slice3 = vgg[6:11]  # conv3_1
        self.slice4 = vgg[11:20]  # conv4_1
        self.slice5 = vgg[20:22]  # conv4_2
        self.slice6 = vgg[22:29]  # conv5_1

        for param in self.parameters():
            param.requires_grad = False

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            features = {}
            x = self.slice1(x)
            features['conv1_1'] = x
            x = self.slice2(x)
            features['conv2_1'] = x
            x = self.slice3(x)
            features['conv3_1'] = x
            x = self.slice4(x)
            features['conv4_1'] = x
            x = self.slice5(x)
            features['conv4_2'] = x
            x = self.slice6(x)
            features['conv5_1'] = x
            return features


def calculate_gram_matrix(x: torch.Tensor) -> torch.Tensor:
    _, channels, height, width = x.size()
    features = x.view(channels, height * width)
    gm = torch.mm(x, x.t())
    return gm / channels * height * width


def run_nst(content_path: str, style_path: str, steps: int = 300,
            content_weight: float = 1, style_weight: float = 1e6) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = load_img(content_path, device=device)
    style_img = load_img(style_path, device=device)

    img_feature_model = ImageFeature().to(device)

    content_features = img_feature_model(content_img)
    style_features = img_feature_model(style_img)

    style_gm = {layer: calculate_gram_matrix(
        style_features[layer]) for layer in style_features if layer != "conv4_2"}

    generated_img = content_img.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([generated_img], lr=0.03)
    style_layer_weights = {'conv1_1': 1.0, 'conv2_1': 0.75,
                           'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}

    for step in range(steps):
        generated_features = img_feature_model(generated_img)
        content_loss = torch.mean(
            (content_features["conv4_2"]-generated_features["conv4_2"])**2)

        style_loss = 0
        for layer, weight in style_layer_weights.items():
            generated_gm = calculate_gram_matrix(generated_features[layer])
            style_loss += weight * \
                torch.mean((style_gm["conv4_2"]-generated_gm["conv4_2"])**2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(
                f"Step [{step}/{steps}] | Total Loss: {total_loss.item():.4f} | Content: {content_loss.item():.4f} | Style: {style_loss.item():.4f}")

    final_image = convert_to_img(generated_img)
    os.makedirs("outputs", exist_ok=True)
    final_image.save("outputs/stylized_output.jpg")
    print("Stylized image saved.")
