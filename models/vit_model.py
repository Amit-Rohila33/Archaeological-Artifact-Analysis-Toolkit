import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class ViTModel(nn.Module):
    def __init__(self, pretrained_model='google/vit-base-patch16-224-in21k'):
        super(ViTModel, self).__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model)
        self.model = AutoModelForImageClassification.from_pretrained(pretrained_model)

    def forward(self, x):
        return self.model(x).logits

def load_vit_model():
    return ViTModel()
