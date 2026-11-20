"""Independent, deterministic received-image transformations."""
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "Pillow", "numpy"]
# ///
# Run through experiment/run_test.py.
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


@dataclass(frozen=True, slots=True)
class Received:
    attack: str
    strength: str
    image: Image.Image
    details: str
    jpeg: bytes | None = None


def crop(image: Image.Image, area: float) -> Image.Image:
    side = int(image.width * math.sqrt(area))
    start = (image.width-side)//2
    return image.crop((start,start,start+side,start+side))


def jpeg(image: Image.Image, quality: int) -> tuple[Image.Image, bytes]:
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality, subsampling=2,
                   optimize=False, progressive=False)
        data = buffer.getvalue()
    with Image.open(io.BytesIO(data)) as decoded:
        return decoded.convert("RGB"), data


def attacks(image: Image.Image) -> Iterator[Received]:
    yield Received("identity", "none", image, "Reopened watermarked PNG")
    for quality in (90,70,50,30):
        output, data = jpeg(image,quality)
        yield Received("jpeg",str(quality),output,"Pillow real JPEG; 4:2:0",data)
    for area in (.5,.25,.1):
        output = crop(image,area)
        yield Received("crop",str(area),output,
                       f"Center crop; retained area; side=int(512*sqrt(area))={output.width}; native decoding")
    for area in (.7,.5,.3):
        side = int(image.width*math.sqrt(area))
        output = image.resize((side,side),Image.Resampling.BICUBIC)
        yield Received("resize",str(area),output,"Retained area ratio; bicubic; native decoding")
    for name, enhancer in (("brightness",ImageEnhance.Brightness),
                           ("contrast",ImageEnhance.Contrast)):
        for factor in (1.5,2.):
            yield Received(name,str(factor),enhancer(image).enhance(factor),"Pillow ImageEnhance")
    for sigma in (.5,1.,2.):
        radius = math.ceil(3*sigma)
        axis = torch.arange(-radius,radius+1,dtype=torch.float32)
        kernel = torch.exp(-axis.square()/(2*sigma*sigma))
        kernel /= kernel.sum()
        value = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()[None]
        value = torch.nn.functional.pad(value,(radius,radius,radius,radius),mode="reflect")
        weight = (kernel[:,None]*kernel[None,:]).expand(3,1,-1,-1)
        value = torch.nn.functional.conv2d(value,weight,groups=3)
        array = value[0].permute(1,2,0).round().clamp(0,255).byte().numpy()
        yield Received("blur",str(sigma),Image.fromarray(array),
                       f"Gaussian sigma pixels; radius {radius}; reflection pad; kernel {kernel.tolist()}")
    yield Received("saturation","2.0",ImageEnhance.Color(image).enhance(2.),"Pillow Color")
    yield Received("sharpness","2.0",ImageEnhance.Sharpness(image).enhance(2.),"Pillow Sharpness")
    yield Received("median","7",image.filter(ImageFilter.MedianFilter(7)),"Pillow median kernel 7×7")
    yield Received("rotation","90",image.transpose(Image.Transpose.ROTATE_90),"Exact counterclockwise 90 degrees")
    text = image.copy()
    ImageDraw.Draw(text).text((64,230),"Stable Signature",fill="white",stroke_width=1,font_size=40)
    yield Received("text","overlay",text,"White Pillow default font, 40px, at (64,230); local explicit variant")
    combined = ImageEnhance.Brightness(crop(image,.5)).enhance(1.5)
    output,data = jpeg(combined,80)
    yield Received("combined","crop.5_bright1.5_jpeg80",output,
                   "Center retained-area .5 → brightness 1.5 → JPEG80; native 362×362",data)
