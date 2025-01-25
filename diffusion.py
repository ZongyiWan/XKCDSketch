import sys
import os
from PIL import Image
import torch
from ip_adapter import IPAdapterXL
import cv2
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "/content/sdxl_models/image_encoder"
ip_ckpt = "/content/sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# load style image
path_to_style_image = "/content/InstantStyle/assets/5.jpg"

style_image = Image.open(path_to_style_image )
style_image.resize((512, 512))


input_image = cv2.imread(path_to_control_image)
detected_map = cv2.Canny(input_image, 50, 200)
canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))
controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)

# load SDXL pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_tiling()
#pipe.enable_sequential_cpu_offload()


# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])