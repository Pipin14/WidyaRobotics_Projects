import os
import numpy as np
import json
from torchvision import transforms

def register_embedding(embedding:np.array, label: str,):
    if not os.path.exists('database.json'):
        database = {}
    else:
        with open('database.json', 'r') as f:database = json.load(f)
    database[label] = embedding.tolist()
    with open('database,json', 'w') as f:
        json.dump(database, f)
      
def cos_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_embedding(embedding:np.array, threshold:float = 0.5):
    if not os.path.exists('database.json'):
        return "Unknown"
    
    with open('database,json', 'r') as f:
        database = json.load(f)
    
    result = []
    
    for label, registered_embedding in database.items():
        similarity = cos_similarity(embedding, np.array(registered_embedding))
        result.append((label, similarity))
    result = sorted(result, key=lambda x: x[1], reverse=True)
    
    if result[0][1] > threshold:
        return result[0][0]
    
    return "Unknown"
