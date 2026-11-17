# /// script
# requires-python = ">=3.7,<3.8"
# dependencies = ["tensorflow==1.15.5", "numpy==1.18.5", "pillow==9.5.0", "bchlib==0.14.0"]
# ///
from pathlib import Path
import hashlib
import os
import json

import bchlib
import numpy as np
from PIL import Image, ImageOps
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import tensorflow.contrib.image

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / '.cache/weights/m1/saved_models/stegastamp_pretrained'
LABEL = 'community mirror (Gu-f), original identity unverified; M1/M3 byte-identical (3/3)'
SEED = 20231130
SECRET = 'Stega!!'
CONFIG = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=8,
                        inter_op_parallelism_threads=1)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def packet() -> np.ndarray:
    bch = bchlib.BCH(137, 5)
    data = bytearray(SECRET, 'utf-8')
    bits = ''.join(format(x, '08b') for x in data + bch.encode(data)) + '0000'
    return np.array([int(x) for x in bits], dtype=np.float32)


def inputs():
    for row in json.loads((ROOT / 'code/input_manifest.json').read_text()):
        path = ROOT / '.cache/inputs' / row['filename']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        with Image.open(str(path)) as source:
            image = np.array(ImageOps.fit(source.convert('RGB'), (400, 400)), np.float32) / 255
        yield path.stem, image


class Network:
    def __init__(self) -> None:
        provenance = json.loads((ROOT / '.cache/weights_provenance.json').read_text())
        assert all(row['identical'] for row in provenance['cross_check'])
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph, config=CONFIG)
        with self.graph.as_default():
            tf.set_random_seed(SEED)
            self.meta = tf.compat.v1.saved_model.loader.load(self.session, ['serve'], str(MODEL))
        sig = self.meta.signature_def['serving_default']
        self.ins = {k: self.graph.get_tensor_by_name(v.name) for k, v in sig.inputs.items()}
        self.outs = {k: self.graph.get_tensor_by_name(v.name) for k, v in sig.outputs.items()}

    def encode(self, image: np.ndarray):
        return self.session.run([self.outs['stegastamp'], self.outs['residual']],
                                {self.ins['image']: [image], self.ins['secret']: [packet()]})

    def decode(self, image: np.ndarray):
        bits = self.session.run(self.outs['decoded'], {self.ins['image']: [image]})[0]
        raw = int(np.sum(bits == packet()))
        binary = ''.join(str(int(x)) for x in bits[:96])
        data = bytearray(int(binary[i:i+8], 2) for i in range(0, 96, 8))
        data, ecc = data[:-5], data[-5:]
        corrections = bchlib.BCH(137, 5).decode_inplace(data, ecc)
        text = None
        if corrections >= 0:
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                text = None
        return {'raw_bits_correct': raw, 'bits': ''.join(str(int(x)) for x in bits),
                'bch_success': corrections >= 0, 'bch_corrections': corrections,
                'recovered_secret': text, 'exact_secret': corrections >= 0 and text == SECRET}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.session.close()
