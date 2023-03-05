from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import base64
import cv2
from typing import List
import numpy as np

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Hello world!'}

@app.post("/convert_image_to_base64")
async def convert_image_to_base64(files: List[UploadFile] = File(...), mode: str = None):
    images = []

    # Check input file
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="The number of input files exceeds the limit, the maximum limit of files that can be input is only 5")

    # Check if file is an image
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File type not supported")

    for file in files:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert the image to grayscale if requested
        if mode == "grayscale":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert the image to negative if requested
        elif mode == "negative":
            img = cv2.bitwise_not(img)

        # Convert the NumPy array to a PIL Image object
        img_pil = Image.fromarray(img)

        # Encode the processed image as base64
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode("ascii")

        images.append(img_str)

    # Return the processed image
    return {
        'mode': mode,
        'images': images,
    }
