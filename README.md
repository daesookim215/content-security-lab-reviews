# Content Security Lab Reviews

Re-implementations and our measurements for six **Research Notes** reviews on
[chickendays.com](https://chickendays.com/). These are our implementations of
the cited methods; we did not find the authors' experimental code. The numeric
results describe our stated settings and are not the papers' original results.

| Paper | Authors | Year | DOI | Folder | Review |
|---|---|---|---|---|---|
| Secure Spread Spectrum Watermarking for Multimedia | I. J. Cox, J. Kilian, F. T. Leighton, T. Shamoon | 1997 | [10.1109/83.650120](https://doi.org/10.1109/83.650120) | [cox-1997-spread-spectrum](cox-1997-spread-spectrum/) | [Blog post](https://chickendays.com/cox-1997-spread-spectrum-watermarking/) |
| Reversible Data Embedding Using a Difference Expansion | J. Tian | 2003 | [10.1109/TCSVT.2003.815962](https://doi.org/10.1109/TCSVT.2003.815962) | [tian-2003-difference-expansion](tian-2003-difference-expansion/) | [Blog post](https://chickendays.com/tian-2003-difference-expansion/) |
| Reversible Data Hiding | Z. Ni, Y.-Q. Shi, N. Ansari, W. Su | 2006 | [10.1109/TCSVT.2006.869964](https://doi.org/10.1109/TCSVT.2006.869964) | [ni-2006-histogram-shifting](ni-2006-histogram-shifting/) | [Blog post](https://chickendays.com/ni-2006-histogram-shifting/) |
| Quantization Index Modulation: A Class of Provably Good Methods for Digital Watermarking and Information Embedding | B. Chen, G. W. Wornell | 2001 | [10.1109/18.923725](https://doi.org/10.1109/18.923725) | [chen-2001-qim](chen-2001-qim/) | [Blog post](https://chickendays.com/chen-2001-qim/) |
| Reversible Watermark Using the Difference Expansion of a Generalized Integer Transform | A. M. Alattar | 2004 | [10.1109/TIP.2004.828418](https://doi.org/10.1109/TIP.2004.828418) | [alattar-2004-integer-transform](alattar-2004-integer-transform/) | [Blog post](https://chickendays.com/alattar-2004-integer-transform/) |
| A Digital Watermark | R. G. van Schyndel, A. Z. Tirkel, C. F. Osborne | 1994 | [10.1109/ICIP.1994.413536](https://doi.org/10.1109/ICIP.1994.413536) | [vanschyndel-1994-digital-watermark](vanschyndel-1994-digital-watermark/) | [Blog post](https://chickendays.com/vanschyndel-1994-digital-watermark/) |

## Quick start

Use Python 3.11. From the repository root:

```sh
python -m pip install -r requirements.txt
python fetch_images.py
python cox-1997-spread-spectrum/reproduce.py
python tian-2003-difference-expansion/reproduce.py
python ni-2006-histogram-shifting/reproduce.py
python chen-2001-qim/reproduce.py
python alattar-2004-integer-transform/reproduce.py
python vanschyndel-1994-digital-watermark/reproduce.py
```

Each reproduction runs the full experiment twice, checks deterministic outputs
and independent decoding, then compares every included result file exactly
against `results/`. It prints PASS or FAIL and exits nonzero on failure. New
images, arrays and metrics go exclusively into each paper's ignored `outputs/`
folder. Recorded `results/` are read-only references. Reproduction can be rerun;
it replaces only its named working runs. Individual run and verification commands
are in the paper READMEs.

Only NumPy, SciPy and Pillow are imported as external dependencies. Floating-point
and JPEG results can depend on the numerical libraries and JPEG backend. The
recorded runs used Python 3.11.15, NumPy 2.3.5, SciPy 1.16.3, Pillow 12.0.0,
libjpeg 6.2 and libjpeg-turbo 3.1.2. Alattar's map compression used zlib 1.2.12;
different compression output can change net capacity. Exact comparisons fail
clearly rather than silently accepting rounded or approximate agreement.

## Test images

Images are not included. `fetch_images.py` downloads USC-SIPI
[Airplane](https://sipi.usc.edu/database/misc/4.2.05.tiff) and
[Boat](https://sipi.usc.edu/database/misc/boat.512.tiff) into ignored `images/`
and verifies fixed SHA-256 hashes. Existing inputs are also verified; a mismatch
fails without replacing the file. Airplane and Boat retain 512x512 dimensions
after Pillow grayscale conversion, except Cox, which uses a center-square crop
and LANCZOS resize to 256x256. Consult the
[USC-SIPI database](https://sipi.usc.edu/database/database.php?volume=misc)
for image provenance and usage terms. USC-SIPI lists these images' source and
copyright status as unknown; downloading them does not grant an MIT image license.

## License

The [MIT license](LICENSE), copyright 2026 daesookim215, applies to the code only.
The cited papers and externally downloaded test images retain their respective
rights. Numeric measurements are provided as experimental records; the MIT
license does not relicense the papers, images or review text.
