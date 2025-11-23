from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from .generator import ImageGen

app = FastAPI()
gen = ImageGen()

class ImageRequest(BaseModel):
    prompt: str
    size: Optional[str] = "1024x1024"
    style: Optional[str] = "photorealistic"
    quality: Optional[str] = "high"
    negative_prompt: Optional[str] = None
    steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    model: Optional[str] = "stable-diffusion"

@app.get("/")
def home():
    return {"message": "Image Generator Backend Running!"}

@app.post("/generate")
def generate(req: ImageRequest):
    result = gen.generate(
        prompt=req.prompt,
        size=req.size,
        style=req.style,
        quality=req.quality,
        negative_prompt=req.negative_prompt,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        model=req.model
    )
    return result
