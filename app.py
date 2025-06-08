from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import io
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

app = FastAPI()

# Carregar o modelo e o processador apenas uma vez
print("🔧 Carregando modelo CLIP...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ Modelo carregado.")

class ImageRequest(BaseModel):
    image_url: str

@app.post("/embed")
async def generate_embedding(req: ImageRequest):
    image_url = req.image_url

    if not image_url or not image_url.startswith('http'):
        return {"error": "⚠️ URL inválida ou não fornecida."}

    try:
        # 1. Baixar a imagem
        response = requests.get(image_url)
        response.raise_for_status()
        image_bytes = io.BytesIO(response.content)

        # 2. Processar a imagem
        imagem = Image.open(image_bytes)
        inputs = processor(images=imagem, return_tensors="pt")

        # 3. Gerar o embedding (512 dimensões)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            embedding_vector_512 = image_features[0].cpu().numpy().tolist()

        # 4. Preencher com zeros para chegar a 768
        embedding_vector_768 = embedding_vector_512 + [0.0] * (768 - len(embedding_vector_512))

        return {
            "embedding": embedding_vector_768,
            "dimensions": len(embedding_vector_768)
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Erro ao baixar a imagem: {e}"}
    except Exception as e:
        return {"error": f"Erro inesperado: {e}"}
