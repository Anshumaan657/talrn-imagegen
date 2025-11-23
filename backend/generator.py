import torch
from diffusers import StableDiffusionPipeline
import time, os, json
from datetime import datetime
from PIL import Image

class ImageGen:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None
        ).to(self.device)

    def generate(
        self,
        prompt,
        negative_prompt=None,
        num_images=1,
        steps=25,
        guidance=7.5,
        seed=None,
        out_dir="samples"
    ):
        os.makedirs(out_dir, exist_ok=True)
        all_meta = []

        for i in range(num_images):
            g = None
            if seed:
                g = torch.manual_seed(seed + i)

            start = time.time()
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=g
            )

            img = output.images[0]
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = f"{timestamp}_{i}.png"
            path = os.path.join(out_dir, filename)

            img.save(path)

            meta = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
                "file": path,
                "timestamp": timestamp,
                "device": self.device
            }

            all_meta.append(meta)

            with open(os.path.join(out_dir, "metadata.jsonl"), "a") as f:
                f.write(json.dumps(meta) + "\n")

        return all_meta
