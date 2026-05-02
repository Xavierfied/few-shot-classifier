import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class ResNetEncoder:
    """
    Extracts 2048-dim embeddings using pretrained ResNet50.
    Final FC Layer is removed - we want features, not class scores.
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            )
        ])


    def encode(self, image_path:str) -> torch.Tensor:
        """Returns L2-normalized embedding of shape (2048,)"""

        img    = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor).squeeze()

        return emb / emb.norm()