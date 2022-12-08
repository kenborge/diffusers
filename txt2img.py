import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

scheduler =  DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base", torch_dtype=torch.float16, revision="fp16", scheduler=scheduler)
pipe = pipe.to("cuda")

torch.manual_seed(69)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
image.show()