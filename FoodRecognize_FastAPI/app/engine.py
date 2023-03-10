import os
import torch
import torch.nn as nn
import torchvision

from typing import Dict, Union
from torchvision import transforms
from quaterion_models.encoders import Encoder

class FoodEncoder(Encoder):
    def __init__(self, encoder_model: nn.Module):
        super().__init__()
        self._encoder = encoder_model
        self._embedding_size = 2048
        
    @property
    def trainable(self) -> bool:
        return False

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def forward(self, image):
        embeddings = self._encoder.forward(image)
        return embeddings
    
    def save(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        torch.save(self._encoder, os.path.join(output_path, 'encoder.pth'))
        raise NotImplementedError("Subclasses should implement this!")
        
    def load(self, input_path: str):
        encoder_model = torch.load(os.path.join(input_path, 'encoder.pth'))
        return FoodEncoder(encoder_model)
        
class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        encoder = torchvision.models.resnet152(weights=None)
        encoder.fc = nn.Identity()
        return FoodEncoder(encoder)

def create_transforms(input_size=336, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transform

def setup_model(checkpoint_path: str = 'foods_recognition/encoder/default'):
    model = Model()
    model = model.configure_encoders()
    model = model.load(checkpoint_path)
    return model

def feature_extractor(image: torch.Tensor, model: nn.Module):
    return model(image)

def preprocess(image: torch.Tensor, transform: transforms.Compose):
    return transform(image).unsqueeze(0)

if __name__ == '__main__':
    model = setup_model()
    input = torch.randn(1, 3, 336, 336)
    output = model(input)
    print(output.shape)