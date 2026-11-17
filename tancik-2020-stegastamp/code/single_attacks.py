# /// script
# requires-python = ">=3.7,<3.8"
# dependencies = ["numpy==1.18.5", "pillow==9.5.0", "opencv-python-headless==4.5.5.64"]
# ///
import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from runtime import SEED


def single_attacks(image: np.ndarray):
    base = Image.fromarray((image * 255).astype(np.uint8))
    yield 'clean', image.copy(), {}
    for quality in (90, 70, 50, 30):
        with io.BytesIO() as buffer:
            base.save(buffer, format='JPEG', quality=quality, subsampling=2, optimize=False)
            buffer.seek(0)
            with Image.open(buffer) as jpeg:
                result = np.array(jpeg.convert('RGB'), np.float32) / 255
        yield 'jpeg_%d' % quality, result, {'quality': quality, 'subsampling': '4:2:0'}
    for sigma in (1, 2):
        result = cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        yield 'blur_%d' % sigma, result, {'sigma_px': sigma, 'kernel': 'OpenCV automatic', 'border': 'reflect101'}
    for std in (.02, .05):
        seed = SEED + int(std * 10000)
        noise = np.random.RandomState(seed).normal(0, std, image.shape).astype(np.float32)
        yield 'noise_%.2f' % std, np.clip(image + noise, 0, 1), {'std': std, 'seed': seed}
    for name, enhancer, values in [('brightness', ImageEnhance.Brightness, (.7, 1.3)),
                                   ('contrast', ImageEnhance.Contrast, (.5, 1.5)),
                                   ('saturation', ImageEnhance.Color, (0, 1.5))]:
        for value in values:
            result = np.array(enhancer(base).enhance(value), np.float32) / 255
            yield '%s_%g' % (name, value), result, {'factor': value, 'implementation': 'Pillow ImageEnhance, uint8'}
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + 18) % 360
    yield 'hue_0.05', np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1), {'turns': .05, 'space': 'float32 HSV'}
    small = base.resize((200, 200), Image.Resampling.LANCZOS)
    yield 'resize_0.5', np.array(small.resize((400, 400), Image.Resampling.LANCZOS), np.float32) / 255, {'scale': .5, 'resampling': 'Lanczos'}
    for degrees in (5, 10):
        rotated = base.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
        yield 'rotation_%d' % degrees, np.array(rotated, np.float32) / 255, {'degrees_ccw': degrees, 'fill': 'black', 'resampling': 'bicubic'}
    src = np.float32([[0, 0], [399, 0], [399, 399], [0, 399]])
    dst = np.float32([[20, 15], [380, 0], [399, 380], [10, 399]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (400, 400), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))
    yield 'perspective_moderate', warped, {'source_corners': src.tolist(), 'destination_corners': dst.tolist(),
                                         'matrix_source_to_destination': matrix.tolist(), 'fill': 'white', 'resampling': 'bilinear'}
    cropped = base.crop((20, 20, 380, 380)).resize((400, 400), Image.Resampling.LANCZOS)
    yield 'crop_0.9', np.array(cropped, np.float32) / 255, {'box': [20, 20, 380, 380], 'resampling': 'Lanczos'}
    yield 'horizontal_flip', image[:, ::-1, :].copy(), {'axis': 'horizontal'}
