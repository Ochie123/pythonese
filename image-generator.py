!pip install torch
import torch
!pip install accelerate
from transformers import pipeline  # Check if transformers can be imported
!pip install diffusers
from diffusers import StableDiffusionPipeline
# Assuming you've installed transformers
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float32
)
#sd_pipe.to("cuda")  # If you have a CUDA-compatible GPU

# Generate an image
prompt = "Generate an image of a bugatti car on sports track ."
image = sd_pipe(prompt).images[0]
image.save("futuristicc.png")