# Run from the repository root: python cox-1997-spread-spectrum/numerical_checks.py
import numpy as np
from scipy.fft import dctn

from imaging import restore_crop, scale
from watermark import ALPHA, LENGTH, embed_float, extract, reference, responses


def numerical_checks() -> dict[str, float]:
    rng = np.random.Generator(np.random.PCG64(424242))
    pixels = rng.uniform(40, 210, (40, 40))
    frequencies = np.arange(40)[:, None]
    samples = np.arange(40)[None, :]
    basis = np.sqrt(2 / 40) * np.cos(np.pi * (2 * samples + 1) * frequencies / 80)
    basis[0] /= np.sqrt(2)
    explicit = basis @ pixels @ basis.T
    owner = reference(pixels)
    transform_error = float(np.max(np.abs(owner.spectrum - explicit)))
    np.testing.assert_allclose(owner.spectrum, explicit, atol=1e-10, rtol=1e-12)
    selected = owner.locations
    assert selected.size == LENGTH and 0 not in selected
    remaining = np.setdiff1d(np.arange(1, pixels.size), selected)
    assert np.min(np.abs(explicit.ravel()[selected])) >= np.max(np.abs(explicit.ravel()[remaining])) - 1e-10
    mark = rng.standard_normal(LENGTH)
    delivered = embed_float(pixels, owner, mark)
    expected = explicit.copy().ravel()
    expected[selected] *= 1 + ALPHA * mark
    expected_image = basis.T @ expected.reshape(pixels.shape) @ basis
    np.testing.assert_allclose(delivered, expected_image, atol=1e-10, rtol=1e-12)
    recovered = extract(delivered, owner)
    recovery_error = float(np.max(np.abs(recovered - mark)))
    np.testing.assert_allclose(recovered, mark, atol=1e-10, rtol=1e-10)
    coefficients = dctn(delivered, type=2, norm='ortho').ravel()
    unchanged = np.setdiff1d(np.arange(pixels.size), selected)
    np.testing.assert_allclose(coefficients[unchanged], explicit.ravel()[unchanged], atol=1e-10)
    score = float(responses(np.array([[2.0, 3.0]]), np.array([3.0, 4.0]))[0])
    np.testing.assert_allclose(score, 18 / 5, atol=1e-14)
    restored = restore_crop(np.full((256, 256), 200.0), np.zeros((256, 256)))
    assert np.count_nonzero(restored) == 128 * 128
    assert float(np.sum(restored)) == 200 * 128 * 128
    assert scale(np.full((256, 256), 100.0)).shape == (256, 256)
    return {'explicit_dct_max_absolute_error': transform_error,
            'float_watermark_max_absolute_recovery_error': recovery_error,
            'independent_detector_formula_score': score}


if __name__ == '__main__':
    print(numerical_checks())
