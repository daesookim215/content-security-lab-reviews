import json
from pathlib import Path

import numpy as np
from PIL import Image

from imaging import ImageSpec, dither, jpeg, prepare, restore_crop, save_image, scale
from records import CASES, PARAMETERS, SEED, Row, parameter_json
from watermark import embed_float, extract, psnr, quantize, reference, responses


def execute_image(spec: ImageSpec, run: Path, seed: int) -> list[Row]:
    name = spec.name
    image_dir = run / 'images' / name
    image_dir.mkdir(parents=True)
    source = prepare(spec, image_dir)
    original = source.original
    (image_dir / 'source.json').write_text(json.dumps({
        'seed': SEED, 'image_seed': seed, 'image': name, 'parameters': PARAMETERS,
        'source_sha256': source.source_sha256,
        'prepared_sha256': source.prepared_sha256, 'source_size': source.source_size,
        'dataset': spec.dataset, 'source_url': spec.url, 'display': spec.display,
    }, indent=2))
    owner = reference(original)
    watermark_seed, candidate_seed = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(2)]
    marks = np.random.Generator(np.random.PCG64(watermark_seed)).standard_normal((5, 1000))
    rng = np.random.Generator(np.random.PCG64(candidate_seed))
    single_bank = rng.standard_normal((1000, 1000))
    single_bank[199] = marks[0]
    multiple_bank = rng.standard_normal((1000, 1000))
    multiple_positions = (99, 299, 499, 699, 899)
    multiple_bank[list(multiple_positions)] = marks
    watermarked_float = embed_float(original, owner, marks[0])
    watermarked = quantize(watermarked_float)
    clipping = float(np.mean((watermarked_float < 0) | (watermarked_float > 255)))
    jpeg10 = jpeg(watermarked, 10, image_dir / 'jpeg10.jpg')
    dithered = dither(watermarked)
    current = original
    clipping_steps = []
    copies = []
    for index, mark in enumerate(marks):
        updated = embed_float(current, owner, mark)
        clipping_steps.append(float(np.mean((updated < 0) | (updated > 255))))
        current = quantize(updated)
        save_image(current, image_dir / f'pass_{index + 1}.png')
        copy = quantize(embed_float(original, owner, mark))
        copies.append(copy)
        save_image(copy, image_dir / f'independent_{index + 1}.png')
    colluded = quantize(np.mean(copies, axis=0))
    attacks = (
        watermarked, scale(watermarked), jpeg10,
        jpeg(watermarked, 5, image_dir / 'jpeg5.jpg'),
        dithered, dithered, restore_crop(watermarked, original),
        restore_crop(jpeg10, original), current, colluded,
    )
    save_image(watermarked[64:192, 64:192], image_dir / 'crop_fragment.png')
    save_image(jpeg10[64:192, 64:192], image_dir / 'jpeg10_crop_fragment.png')
    scaled_down = Image.fromarray(watermarked.astype(np.uint8)).resize((128, 128), Image.Resampling.LANCZOS)
    save_image(np.asarray(scaled_down, dtype=np.float64), image_dir / 'scaled_128.png')
    (image_dir / 'clipping.json').write_text(json.dumps({
        'seed': SEED, 'image': name, 'parameters': PARAMETERS,
        'first_insertion': clipping, 'successive_insertions': clipping_steps,
    }, indent=2))
    np.savez_compressed(image_dir / 'secrets.npz', marks=marks,
                        single_candidates=single_bank, multiple_candidates=multiple_bank,
                        locations=owner.locations, original_coefficients=owner.spectrum.ravel()[owner.locations])
    rows = []
    for case, attacked in zip(CASES, attacks, strict=True):
        multiple = case.name in ('rewatermark', 'collusion')
        positions = multiple_positions if multiple else (199,)
        bank = multiple_bank if multiple else single_bank
        recovered = extract(attacked, owner)
        if case.postprocess == 'subtract mean':
            recovered = recovered - np.mean(recovered)
        scores = responses(bank, recovered)
        fidelity = psnr(original, attacked)
        np.save(image_dir / f'{case.name}_extracted.npy', recovered)
        save_image(attacked, image_dir / f'{case.name}.png')
        for index, score in enumerate(scores):
            rows.append(Row(run.name, SEED, name, spec.dataset, seed, watermark_seed, candidate_seed,
                            case.name, parameter_json(case), index, index in positions,
                            positions.index(index) if index in positions else -1,
                            float(score), bool(score > 6), fidelity,
                            float(np.mean(recovered)), float(np.linalg.norm(recovered)),
                            clipping_steps[-1] if case.name == 'rewatermark' else clipping,
                            source.source_sha256, source.prepared_sha256))
    print(f'{name}: finished {len(CASES)} cases, {len(rows)} detector responses', flush=True)
    return rows
