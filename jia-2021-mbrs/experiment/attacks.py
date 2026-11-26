# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "torchvision==0.29.2", "kornia==0.9.2", "Pillow==13.0.0", "numpy==2.4.6"]
# ///
# Run: imported by run_test.py after official modules are loaded
"""Independent attacks on a serialized watermark using official noise definitions."""
from io import BytesIO
from pathlib import Path
from typing import Final
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

NORMALIZE: Final = transforms.Compose([transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])

def as_tensor(image: Image.Image) -> torch.Tensor:
    return NORMALIZE(image).unsqueeze(0).float()

def png_image(tensor: torch.Tensor) -> Image.Image:
    pixels=((tensor[0].clamp(-1,1).permute(1,2,0)+1)*255/2).round().to(torch.uint8).numpy()
    return Image.fromarray(pixels)

def jpeg(image: Image.Image, quality: int) -> torch.Tensor:
    with BytesIO() as stream:
        image.save(stream,format='JPEG',quality=quality,subsampling=2)
        stream.seek(0)
        with Image.open(stream) as received:
            return as_tensor(received.convert('RGB'))

def generate(saved: Path, cover: torch.Tensor):
    from network.noise_layers import GF, GN, MF, SP, Crop, Cropout, Dropout
    with Image.open(saved) as opened:
        image=opened.convert('RGB')
    base=as_tensor(image)
    yield 'encoded_png','none',base
    for quality in (90,70,50,30,10):
        yield 'jpeg',f'Q{quality}',jpeg(image,quality)
    for name,strength,layer in [('gaussian_filter','sigma=2; kernel=7',GF(2)),('gaussian_noise','variance=0.01 in [-1,1]',GN(.01)),('median_filter','kernel=3',MF(3)),('salt_pepper','p=0.01',SP(.01)),('crop','side=0.19; area~=0.035',Crop(.19,.19)),('cropout','retained_area=0.3',Cropout(np.sqrt(.3),np.sqrt(.3))),('dropout','retained_probability=0.3',Dropout(.3))]:
        yield name,strength,layer([base.clone(),cover])
    small=torch.nn.functional.interpolate(base,scale_factor=.5,mode='bilinear',align_corners=False,antialias=True)
    restored=torch.nn.functional.interpolate(small,size=base.shape[-2:],mode='bilinear',align_corners=False,antialias=True)
    yield 'resize','side=0.5 then restore',restored
