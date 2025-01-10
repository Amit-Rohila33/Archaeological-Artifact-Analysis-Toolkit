from models.vit_model import load_vit_model
from pipeline.preprocess import preprocess_dataset
import torch
import numpy as np

def segment_artifacts(image_dir):
    vit_model = load_vit_model()
    processed_images = preprocess_dataset(image_dir)

    with torch.no_grad():
        predictions = []
        for img in processed_images:
            img_tensor = torch.tensor(img).permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
            prediction = vit_model(img_tensor)
            predictions.append(prediction.numpy())
    return np.array(predictions)
