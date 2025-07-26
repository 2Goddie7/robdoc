import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

def train_model(rutaDatos='datos/salud_datos.csv'):
    print("Ruta actual:", os.getcwd())
    print("Verificando existencia del archivo:", rutaDatos)
    if not os.path.exists(rutaDatos):
        print("El archivo no existe en esa ruta.")
        return

    data = pd.read_csv(rutaDatos, encoding='utf-8', quotechar='"')
    print("CSV cargado correctamente.")
    print("Columnas encontradas:", data.columns)

    model_embed = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model_embed.encode(data['question'].tolist(), convert_to_numpy=True)

    model = {
        'embedding_model_name': 'all-MiniLM-L6-v2',
        'matrix': embeddings,
        'questions': data['question'].tolist(),
        'answers': data['answer'].tolist(),
    }

    rutaGuardado = 'modelo/model.pkl'
    print("Guardando modelo en:", rutaGuardado)
    with open(rutaGuardado, 'wb') as f:
        pickle.dump(model, f)

    print("Modelo entrenado y guardado exitosamente.")
    return model

if __name__ == '__main__':
    train_model()