import torch

try:
    # Package import path when called from project root.
    from .encoder import ResNetEncoder
except ImportError:
    # Fallback for direct script execution.
    from encoder import ResNetEncoder

class RoomClassifier:
    def __init__(self, embeddings_path= "embeddings/support.pt"):
        self.encoder = ResNetEncoder()
        self.prototypes = torch.load(embeddings_path, map_location=self.encoder.device)
        self.labels = list(self.prototypes.keys())

        # Stack into matrix: shape (num_classes, 2048)
        self.protomatrix = torch.stack(list(self.prototypes.values()))

    def predict(self, image_path):
        query_emb = self.encoder.encode(image_path)

        similarities = (self.protomatrix @ query_emb).tolist()
        scores       = dict(zip(self.labels, similarities))
        return max(scores, key=scores.get), scores