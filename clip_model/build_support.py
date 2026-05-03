from clip_fewshot import CLIPFewShotClassifier

clf = CLIPFewShotClassifier()
clf.build_support("support_images/", "embeddings/clip_support.pt")