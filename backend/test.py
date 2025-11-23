import sys, os

# Make Python find the backend folder
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from backend.generator import ImageGen

print("Import worked! 🔥")

gen = ImageGen()
meta = gen.generate("a cute robot holding a flower", num_images=1)

print("Image generated!")
print(meta)
