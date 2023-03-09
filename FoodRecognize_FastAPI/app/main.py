import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from utils import register_embedding, recognize_embedding
from engine import create_transforms, setup_model, feature_extractor, preprocess
import io


app = FastAPI()


model = setup_model()
transforms = create_transforms()


@app.post("/register")
async def register_embedding(label: str = '', image: UploadFile = File(...), ):
    image_bytes = await image.read()
    image_stream = io.BytesIO(image_bytes)
    
    image = Image.open(image_stream).convert('RGB')
    image = transforms(image)
    image = image.unsqueeze(0)
    
    embedding = model(image)
    label = register_embedding(label, embedding.detach().squeeze().numpy())
    
    return {"message": "Successfully registered"}   


@app.post("/recognize")
async def recognize(image: UploadFile = File(...), threshold: float=0.5):
    image_bytes = await image.read()
    image_stream = io.BytesIO(image_bytes)
    
    image = Image.open(image_stream).convert('RGB')
    image = transforms(image)
    image = image.unsqueeze(0)
    
    embedding = model(image)
    label = recognize_embedding(embedding.detach().squeeze().numpy(), threshold=threshold)
    
    return {"label": label}