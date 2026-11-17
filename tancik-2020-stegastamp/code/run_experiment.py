# /// script
# requires-python = ">=3.7,<3.8"
# dependencies = ["tensorflow==1.15.5", "numpy==1.18.5", "pillow==9.5.0", "bchlib==0.14.0", "scikit-image==0.19.3", "opencv-python-headless==4.5.5.64"]
# ///
import csv
import hashlib
import json
import argparse
import time
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from runtime import ROOT, LABEL, SEED, SECRET, Network, inputs, packet, write_json
from single_attacks import single_attacks
from combined_attacks import Combined


def checksum(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def statistics(array: np.ndarray):
    return {'min': float(array.min()), 'max': float(array.max()),
            'mean': float(array.mean()), 'std': float(array.std()),
            'mean_abs': float(np.abs(array).mean()), 'rms': float(np.sqrt(np.mean(array ** 2)))}


def main() -> None:
    start = time.monotonic()
    cv2.setNumThreads(1)
    np.random.seed(SEED)
    parser = argparse.ArgumentParser(description='Run the fixed CPU experiment.')
    parser.add_argument('run', choices=('run1', 'run2'))
    destination = ROOT / 'run-output' / parser.parse_args().run
    destination.mkdir(parents=True, exist_ok=True)
    quality, rows, combined = [], [], []
    with Network() as net, Combined() as chain:
        loaded = time.monotonic()
        for index, (name, image) in enumerate(inputs()):
            hidden, residual = net.encode(image)
            hidden, residual = hidden[0], residual[0]
            encoded_u8 = (hidden * 255).astype(np.uint8)
            encoded = encoded_u8.astype(np.float32) / 255
            np.save(str(destination / (name + '_encoded_float.npy')), hidden, allow_pickle=False)
            np.save(str(destination / (name + '_residual_float.npy')), residual, allow_pickle=False)
            np.save(str(destination / (name + '_encoded_uint8.npy')), encoded_u8, allow_pickle=False)
            quality.append({'label': LABEL, 'image': name,
                'psnr_db': float(peak_signal_noise_ratio(image, encoded, data_range=1)),
                'ssim': float(structural_similarity(image, encoded, data_range=1, channel_axis=2)),
                'ssim_settings': 'skimage 0.19.3, win_size=7, uniform weights, sample covariance, RGB channel mean',
                'float_psnr_db': float(peak_signal_noise_ratio(image, hidden, data_range=1)),
                'float_ssim': float(structural_similarity(image, hidden, data_range=1, channel_axis=2)),
                'network_residual': statistics(residual), 'saved_png_delta': statistics(encoded - image),
                'input_sha256_float32': checksum(image), 'encoded_sha256_uint8': checksum(encoded_u8),
                'residual_sha256_float32': checksum(residual),
                'clipped_pixel_channels': int(np.count_nonzero((image + residual < 0) | (image + residual > 1)))})
            for attack, changed, params in single_attacks(encoded):
                assert changed.shape == (400, 400, 3) and np.isfinite(changed).all()
                recovery = net.decode(changed)
                if attack == 'clean':
                    assert recovery['exact_secret'], recovery
                rows.append({'label': LABEL, 'image': name, 'attack': attack,
                             'attack_params': params, 'attacked_tensor_sha256': checksum(changed), **recovery})
                print(name, attack, recovery['raw_bits_correct'], recovery['exact_secret'], flush=True)
            for draw in range(20):
                seed = SEED + index * 1000 + draw
                changed, params = chain.draw(encoded, seed)
                assert changed.shape == (400, 400, 3) and np.isfinite(changed).all()
                recovery = net.decode(changed)
                combined.append({'label': LABEL, 'image': name, 'draw': draw + 1, 'params': params,
                                 'attacked_tensor_sha256': checksum(changed), **recovery})
                print(name, 'combined', draw + 1, recovery['raw_bits_correct'], recovery['exact_secret'], flush=True)
    inference_complete = time.monotonic()
    summary = []
    for name, _ in inputs():
        subset = [row for row in combined if row['image'] == name]
        summary.append({'label': LABEL, 'image': name, 'draws': len(subset),
                        'exact_recovery_count': sum(row['exact_secret'] for row in subset),
                        'bch_success_count': sum(row['bch_success'] for row in subset),
                        'mean_bits': float(np.mean([row['raw_bits_correct'] for row in subset])),
                        'min_bits': min(row['raw_bits_correct'] for row in subset)})
    write_json(destination / 'quality.json', quality)
    write_json(destination / 'single_attacks.json', rows)
    write_json(destination / 'combined_draws.json', combined)
    write_json(destination / 'combined_summary.json', summary)
    write_json(destination / 'settings.json', {'label': LABEL, 'seed': SEED, 'secret': SECRET,
               'packet_bits': ''.join(str(int(x)) for x in packet()),
               'attack_input': 'official uint8 truncation encoded PNG, reloaded as float32/255',
               'combined_order': 'perspective white fill -> SAME zero-pad blur -> noise/clip -> contrast+RGB/brightness offsets/clip -> desaturation -> official JPEG surrogate'})
    for filename, data in [('quality', quality), ('single_attacks', rows),
                           ('combined_draws', combined), ('combined_summary', summary)]:
        with (destination / (filename + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            for row in data:
                writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in row.items()})
    write_json(destination / 'timing.json', {'total_seconds': time.monotonic() - start,
               'model_and_chain_load_seconds': loaded - start, 'inference_and_attacks_seconds': inference_complete - loaded})


if __name__ == '__main__':
    main()
