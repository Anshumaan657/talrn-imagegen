import torch
from diffusers import StableDiffusionPipeline
import time, os, json
from datetime import datetime
from PIL import Image

class ImageGen:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔥 Loading model on: {self.device}")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None
        ).to(self.device)

    def generate(
        self,
        prompt,
        size="1024x1024",
        style="photorealistic",
        quality="high",
        negative_prompt=None,
        steps=30,
        guidance_scale=7.5,
        seed=None,
        model="stable-diffusion",
        out_dir="samples"
    ):
        os.makedirs(out_dir, exist_ok=True)

        # Parse size (e.g., "768x768")
        try:
            width, height = map(int, size.lower().split("x"))
        except:
            width, height = 512, 512  # fallback

        # Style presets
        STYLE_PRESETS = {
            "photorealistic": "hyper-realistic, 8k, RAW, ultra details",
            "anime": "anime style, detailed, colorful, crisp lineart",
            "digital-art": "digital painting, vibrant lighting, concept art",
            "cinematic": "cinematic lighting, film grain, dramatic"
        }

        if style in STYLE_PRESETS:
            prompt = f"{prompt}, {STYLE_PRESETS[style]}"

        # Quality presets
        QUALITY_PRESETS = {
            "low": 15,
            "medium": 30,
            "high": 50
        }

        if quality in QUALITY_PRESETS:
            steps = QUALITY_PRESETS[quality]

        generator = None
        if seed:
            generator = torch.manual_seed(seed)

        # Generate
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        img = output.images[0]

        # Save image
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{timestamp}.png"
        path = os.path.join(out_dir, filename)
        img.save(path)

        # Metadata
        meta = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "size": size,
            "model": model,
            "style": style,
            "quality": quality,
            "device": self.device,
            "file": path,
            "timestamp": timestamp
        }

        # Save metadata log
        with open(os.path.join(out_dir, "metadata.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")

        return meta
