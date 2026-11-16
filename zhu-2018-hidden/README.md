# HiDDeN released-checkpoint image tests

CPU inference for the HiDDeN review. No training, images, checkpoints or paper files are included in this folder.

Run with Python 3.11 and uv:

```sh
uv run reproduce.py
```

If the Python process has restricted network access and Ego Lite is available,
use `uv run reproduce.py --fetch-with-ego`. That route downloads the same archive
and images through Ego Lite's Node runtime, verifies their hashes, then runs
the same local CPU inference. This fallback requires a working Ego Lite CLI.

The script downloads the Ando Khachatryan implementation at commit
`556f76dd0602e4351ed4c19bd2ee87d50ef3c0de` from GitHub's source archive.
It verifies the archive SHA-256 and both checkpoint hashes before loading tensor
states with `weights_only=True`. It downloads Airplane and Boat from USC-SIPI
and verifies their SHA-256 hashes. Downloads remain in the ignored `.cache/`
folder. It never invokes git.

The test uses central 128×128 RGB crops, fixed message
`101100101100101100101100101100`, float32 CPU evaluation, PNG serialization,
real JPEG, center and seeded random crops, cover-substitution dropout and
Gaussian blur. Settings and all decoder predictions are in `results/`.
`results/measurements.csv` has 612 per-received-image rows. `quality.csv`
records encoded-image RGB PSNR against each cover, raw-float PSNR and clipping.
Crop rows carry the encoded-image PSNR as an embedding-quality property, not
PSNR between a crop and a full image.

`run-output/` contains newly generated images, masks, tensors and numeric files.
The reproduction report compares five result files byte for byte with the
recorded run, and `settings.json` field by field except the Python patch version. CPU, package versions and backend can affect floating point
values on other systems; bit recovery is thresholded at ≥0.5, without sigmoid.
The exact comparison was performed with the versions in `results/settings.json`.

Dropout uses twenty seeded Bernoulli masks per strength, shared across models.
Random-location crop has twenty seeded boxes per strength; center crops remain
separate. All predictions and draws are retained. `summary.json` gives per-image,
model, family and strength mean/minimum/maximum correct bits, with denominator 30.

Source code: [Ando implementation, MIT](https://github.com/ando-khachatryan/HiDDeN/tree/556f76dd0602e4351ed4c19bd2ee87d50ef3c0de);
[authors' Lua implementation, MIT](https://github.com/jirenz/HiDDeN/tree/9baa4a79dffe293f7553068f92eb7527534cba26).
The review runner and transformations are in `code/`; the downloaded modules
are unmodified.

Data: [USC-SIPI](https://sipi.usc.edu/database/), miscellaneous Airplane
`4.2.05.tiff` and Boat `boat.512.tiff`. USC-SIPI lists their copyright as
unknown in its [copyright notes](https://sipi.usc.edu/database/copyright.php).
The images are fetched for the test and are not redistributed here.

Paper: Zhu et al., [HiDDeN: Hiding Data With Deep Networks](https://arxiv.org/abs/1807.09937v1), 2018.
