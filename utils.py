from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

class Models:
    
    @staticmethod
    def emotion_model():
        return pipeline("text-classification", model="michellejieli/emotion_text_classifier", device=device, tokenizer="michellejieli/emotion_text_classifier", truncation=True)