from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

with open('modelo/model.pkl', 'rb') as f:
    model = pickle.load(f)

embed_model = SentenceTransformer(model['embedding_model_name'])

class Consulta(BaseModel):
    pregunta: str

@app.post("/consultar")
def consultar_salud(consulta: Consulta):
    user_vec = embed_model.encode([consulta.pregunta])
    sim = cosine_similarity(user_vec, model['matrix']).flatten()
    idx = sim.argmax()
    return {"respuesta": model['answers'][idx]}