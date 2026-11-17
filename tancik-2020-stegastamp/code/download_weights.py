# /// script
# requires-python = ">=3.7"
# dependencies = []
# ///
import filecmp
import json
from pathlib import PurePosixPath
import shutil
import stat
import zipfile

from download_file import CACHE, Download, digest, fetch

ROOT = CACHE
M1 = 'https://media.githubusercontent.com/media/Gu-f/StegaStamp/d95ea307c73410c96557200656d2e0a59574eedb/saved_models/stegastamp_pretrained/'
M3 = 'https://raw.githubusercontent.com/ytfrdfiw/StegaStamp/84e34ef46e27b5fe847d474853a133b730a1ec03/saved_models/'
FILES = [
    ('saved_model.pb', 241765, '42c4be663021a1d98f82eed425ac6643aa149d57e8bb5a60be2247e19937f095'),
    ('variables/variables.data-00000-of-00001', 217018916, '1ca5cd940906f57454fcf3d05d4f81edd87066903f7ced2ba57399614bca3bf0'),
    ('variables/variables.index', 2416, '8c038f8fa37dba430c59dd91a654f2aedffff27acf0efed657d2d15f46d5149c'),
]


class WeightVerificationError(RuntimeError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


def main() -> None:
    records = []
    jobs = [(M1 + name, ROOT / 'weights/m1/saved_models/stegastamp_pretrained' / name, size, sha)
            for name, size, sha in FILES]
    part_hashes = [
        'b01fc5b39042965fccfac8149539e7fccca115ae565f2c426ef45e2d531cdb05',
        '23f3054b3cb76337d2c9748ff2410aba09c282c370eadaf1d2df24aa88a1462b',
        '0a48d3c68f5836e181b01d828abe28e20a716e9153593dce226b65c80e10bdab',
    ]
    for i, size in enumerate([100000000, 100000000, 17264075], 1):
        name = 'stegastamp_pretrained.zip.%03d' % i
        jobs.append((M3 + name, ROOT / 'weights/m3' / name, size,
                     part_hashes[i-1]))
    for url, path, size, expected in jobs:
        path.parent.mkdir(parents=True, exist_ok=True)
        fetch(Download(url, path, size, expected))
        sha = digest(path)
        used = url
        records.append({'url': used, 'requested_url': url, 'path': str(path.relative_to(ROOT)),
                        'size': size, 'sha256': sha})
        print(json.dumps(records[-1]), flush=True)
    archive = ROOT / 'weights/m3/stegastamp_pretrained.zip'
    with archive.open('wb') as output:
        for i in range(1, 4):
            with (ROOT / ('weights/m3/stegastamp_pretrained.zip.%03d' % i)).open('rb') as source:
                shutil.copyfileobj(source, output)
    if (archive.stat().st_size != 217264075
            or digest(archive) != 'cd4aeb5dcd7c97be8d0907143d7e5103ea9f3b9e509490f0cf20f67bf6d8bcde'):
        raise WeightVerificationError('Concatenated archive checksum mismatch')
    with zipfile.ZipFile(str(archive)) as z:
        listing = []
        for entry in z.infolist():
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in entry.filename
                    or not path.parts or path.parts[0] != 'stegastamp_pretrained'
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise WeightVerificationError('Rejected archive entry: ' + entry.filename)
            listing.append({'name': entry.filename, 'size': entry.file_size,
                            'external_attr': entry.external_attr})
        (ROOT / 'weights/m3/archive_listing.json').write_text(json.dumps(listing, indent=2) + '\n')
        print(json.dumps(listing, indent=2), flush=True)
        z.extractall(str(ROOT / 'weights/m3/extract'))
    comparisons = []
    for name, size, sha in FILES:
        a = ROOT / 'weights/m1/saved_models/stegastamp_pretrained' / name
        b = ROOT / 'weights/m3/extract/stegastamp_pretrained' / name
        code = 0 if filecmp.cmp(str(a), str(b), shallow=False) else 1
        comparisons.append({'file': name, 'm1_sha256': digest(a), 'm3_sha256': digest(b),
                            'm3_size': b.stat().st_size, 'cmp_exit': code,
                            'identical': code == 0 and digest(b) == sha})
    result = {'label': 'community mirror (Gu-f), original identity unverified',
              'downloads': records, 'archive_sha256': digest(archive),
              'archive_size': archive.stat().st_size, 'cross_check': comparisons}
    (ROOT / 'weights_provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2), flush=True)
    if not all(item['identical'] for item in comparisons):
        raise WeightVerificationError('Mirrors differ; numerical comparison required before experiments')


if __name__ == '__main__':
    main()
