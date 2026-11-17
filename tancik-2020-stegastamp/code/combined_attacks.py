# /// script
# requires-python = ">=3.7,<3.8"
# dependencies = ["tensorflow==1.15.5", "numpy==1.18.5", "opencv-python-headless==4.5.5.64"]
# ///
# Imported by run_experiment.py; uses the unchanged official JPEG implementation.
import hashlib
import sys
import cv2
import numpy as np
from runtime import ROOT, CONFIG, tf

sys.path.insert(0, str(ROOT / '.cache/official-source'))
import utils


class Combined:
    def __init__(self) -> None:
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.image = tf.placeholder(tf.float32, [1, 400, 400, 3])
            self.matrix = tf.placeholder(tf.float32, [1, 8])
            self.kernel = tf.placeholder(tf.float32, [7, 7, 3, 3])
            self.noise = tf.placeholder(tf.float32, [1, 400, 400, 3])
            self.contrast = tf.placeholder(tf.float32, [])
            self.offset = tf.placeholder(tf.float32, [1, 1, 1, 3])
            self.saturation = tf.placeholder(tf.float32, [])
            self.factor = tf.placeholder(tf.float32, [])
            warped = tf.contrib.image.transform(self.image, self.matrix, interpolation='BILINEAR')
            mask = tf.contrib.image.transform(tf.ones_like(self.image), self.matrix, interpolation='BILINEAR')
            warped += 1 - mask
            result = tf.nn.conv2d(warped, self.kernel, [1, 1, 1, 1], padding='SAME')
            result = tf.clip_by_value(result + self.noise, 0, 1)
            result = tf.clip_by_value(result * self.contrast + self.offset, 0, 1)
            lum = tf.reduce_sum(result * tf.constant([.3, .6, .1]), axis=3, keepdims=True)
            result = (1 - self.saturation) * result + self.saturation * lum
            self.result = utils.jpeg_compress_decompress(result, rounding=utils.round_only_at_0,
                                                        factor=self.factor, downsample_c=True)
        self.session = tf.Session(graph=self.graph, config=CONFIG)

    def draw(self, image: np.ndarray, seed: int):
        rng = np.random.RandomState(seed)
        bound = int(np.floor(400 * .1 * rng.uniform()))
        offsets = rng.uniform(-bound, bound, (4, 2)).astype(np.float32)
        src = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
        dst = src + offsets
        matrix = np.linalg.inv(cv2.getPerspectiveTransform(src, dst))
        matrix /= matrix[2, 2]
        sigma_g = rng.uniform(1, 3)
        angle = rng.uniform(0, 2 * np.pi)
        sigma_line = rng.uniform(.25, 1)
        width_line = rng.uniform(3, 3.1)
        choice = rng.uniform()
        coords = np.stack(np.meshgrid(np.arange(7), np.arange(7), indexing='ij'), -1).astype(np.float32) - 3
        manhattan = np.sum(np.abs(coords), -1)
        gaussian = np.exp(-np.sum(coords ** 2, -1) / (2 * sigma_g ** 2))
        distance = np.sum(coords * np.float32([np.cos(angle), np.sin(angle)]), -1)
        motion = np.exp(-distance ** 2 / (2 * sigma_line ** 2)) * (manhattan < width_line)
        weights = (manhattan < .5).astype(np.float32)
        if choice < .5:
            weights = motion
        if choice < .25:
            weights = gaussian
        weights /= weights.sum()
        kernel = np.zeros((7, 7, 3, 3), np.float32)
        for c in range(3):
            kernel[:, :, c, c] = weights
        noise_std = rng.uniform(0, .02)
        noise_seed = int(rng.randint(0, 2 ** 31 - 1))
        noise = np.random.RandomState(noise_seed).normal(0, noise_std, (1, 400, 400, 3)).astype(np.float32)
        contrast = rng.uniform(.5, 1.5)
        hue_offsets = rng.uniform(-.1, .1, (1, 1, 1, 3)).astype(np.float32)
        brightness = rng.uniform(-.3, .3)
        saturation = rng.uniform(0, 1)
        quality = rng.uniform(50, 100)
        factor = (200 - 2 * quality) / 100 + .0001
        feed = {self.image: [image], self.matrix: matrix.flatten()[:8].reshape(1, 8),
                self.kernel: kernel, self.noise: noise, self.contrast: contrast,
                self.offset: hue_offsets + brightness, self.saturation: saturation, self.factor: factor}
        result = self.session.run(self.result, feed)[0]
        params = {'seed': seed, 'corner_bound_px': bound, 'corner_offsets': offsets.tolist(),
                  'matrix_output_to_input': matrix.tolist(), 'blur_selector': choice,
                  'blur_kind': ['gaussian', 'motion', 'identity'][int(choice >= .25) + int(choice >= .5)],
                  'gaussian_sigma': sigma_g, 'motion_angle_radians': angle,
                  'motion_sigma': sigma_line, 'motion_manhattan_threshold': width_line,
                  'kernel_sha256': hashlib.sha256(kernel.tobytes()).hexdigest(),
                  'noise_std': noise_std, 'noise_seed': noise_seed,
                  'noise_sha256': hashlib.sha256(noise.tobytes()).hexdigest(),
                  'contrast': contrast, 'rgb_offsets': hue_offsets.flatten().tolist(),
                  'brightness_offset': brightness, 'desaturation': saturation,
                  'jpeg_quality': quality, 'jpeg_factor': factor}
        return result, params

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.session.close()
