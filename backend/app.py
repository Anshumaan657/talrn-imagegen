from fastapi import FastAPI
from pydantic import BaseModel
from generator import generate_image

app = FastAPI()

class ImageRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "Image Generator Backend Running!"}

@app.post("/generate")
def generate(req: ImageRequest):
    return generate_image(req.prompt)
