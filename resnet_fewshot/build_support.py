import os
import torch

try:
    # Package import path when called from project root.
    from .encoder import ResNetEncoder
except ImportError:
    # Fallback for direct script execution.
    from encoder import ResNetEncoder

def build_support_embeddings(support_dir, output_path="embeddings/support.pt"):
    """
    Walk support_dir, encode each image, save prototype embeddings.
    Run this ONCE after adding support images.

    Folder Structure:
        support_dir:
            kitchen/img1.jpg, img2.jpg
            bedroom/img1.jpg
    """

    encoder = ResNetEncoder()
    prototypes = {}

    for label in os.listdir(support_dir):
        class_path = os.path.join(support_dir, label)
        if not os.path.isdir(class_path): continue

        embeddings= []
        for fname in os.listdir(class_path):
            emb = encoder.encode(os.path.join(class_path, fname))
            embeddings.append(emb)

        if embeddings:
            proto = torch.stack(embeddings).mean(dim=0)
            prototypes[label] = proto / proto.norm()
            print(f"    {label}: {len(embeddings)} images")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(prototypes, output_path)


if __name__ == "__main__":
    build_support_embeddings("support_images/", "embeddings/support.pt")
    print("Files Embedded!")