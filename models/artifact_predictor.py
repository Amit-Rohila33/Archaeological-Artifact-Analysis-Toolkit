import torch
from transformers import BertTokenizer, BertForSequenceClassification

class ArtifactPredictor:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def predict(self, metadata_text):
        inputs = self.tokenizer(metadata_text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()  # Class prediction
