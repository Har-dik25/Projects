import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)
pipe.to("cpu")


def generate_frames(prompt: str, num_frames: int = 8):
    frames = []

    for i in range(num_frames):
        img = pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.0
        ).images[0]

        img = img.resize((384, 384))
        frames.append(np.array(img))

    return frames
