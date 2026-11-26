# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.15.0", "torchvision==0.29.2", "kornia==0.9.2", "Pillow==13.0.0", "numpy==2.4.6", "scipy==1.17.1", "scikit-image==0.26.0", "pypdfium2==5.15.2"]
# ///
# Run: uv run reproduce.py --local-inputs /path/to/jia-2021-mbrs
"""Fetch hash-pinned inputs or use offline inputs, run twice, compare exact results."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile
from typing import Final

HERE: Final = Path(__file__).resolve().parent
COMMIT: Final = '2bab0df66480f66cfdef534d38c851354ee2c6aa'
ARCHIVE_SHA256: Final = '390d83b8cc1d5c168fa8f664c15f281949a3671f15fe2d88eb8d54712d3dbf18'
WEIGHTS: Final = (
 ('MBRS_256_m256/EC_42.pth','13R8F_DsmC7firokQ5lSBTkgiuK1naryc','6b49e01b85fbef7356a93fa3914a396e10f53e9dca2a745eaef578173ef999c5'),
 ('MBRS_256_m256/D_42.pth','1YAs5-KXhEy30WQOgWxoYL8jJyCWx_yL2','d89198cd002cb86064fb10eb26b0416ccb6ab90cf1a26f46d61bbd66d3945432'),
 ('MBRS_Diffusion_128_m30/EC_114.pth','1JSUs17jkc58i2m0bxF74XQSN-a1ekANY','53d89e1199cd976bd5bf8d7c99166a94fbbed66dc4a712cb3b17ca2477f1eb63'),
 ('MBRS_Diffusion_128_m30/D_114.pth','1wkCSLtTof5KOii1Uy_nqFOBCLxuwdmiP','a2c0c198a33e49840b554478d9237b22c393ed123cda57a13caf824ad1acf6e1'),
)
IMAGES: Final = (
 ('4.2.05.tiff','ae9eba12b80df955e47585bf4a16332ea0dadc753668d3457c509564934dbd40'),
 ('boat.512.tiff','6bc7319b1e3b44ce0b2a571ac39a28bac94437d812057d1afb2cf78ea3a9a35f'),
)
FILES: Final = ('results.json','measurements.csv','quality.json','quality.csv','settings.json','example_values.json')


def download(url: str, destination: Path, expected: str) -> None:
    """Download via stdlib urllib and accept only the expected SHA-256."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==expected:
        return
    request=urllib.request.Request(url,headers={'User-Agent':'MBRS-review-reproduction/1.0'})
    with urllib.request.urlopen(request,timeout=120) as response:
        payload=response.read()
    actual=hashlib.sha256(payload).hexdigest()
    assert actual==expected,(destination.name,expected,actual)
    destination.write_bytes(payload)


def prepare(inputs: Path) -> None:
    """Create an input cache without redistributing unlicensed official sources."""
    assert len(ARCHIVE_SHA256)==64 and all(c in '0123456789abcdef' for c in ARCHIVE_SHA256), 'Fill ARCHIVE_SHA256 with the independently obtained archive hash before using downloads; --local-inputs works offline.'
    archive=inputs/'source.zip'
    download(f'https://github.com/jzyustc/MBRS/archive/{COMMIT}.zip',archive,ARCHIVE_SHA256)
    source=inputs/'code/MBRS'
    if not source.exists():
        with zipfile.ZipFile(archive) as zipped:
            prefix=f'MBRS-{COMMIT}/'
            for member in zipped.infolist():
                assert member.filename.startswith(prefix),member.filename
                relative=Path(member.filename[len(prefix):])
                assert not relative.is_absolute() and '..' not in relative.parts,member.filename
                target=source/relative
                if member.is_dir():
                    target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(zipped.read(member))
    for filename,drive_id,checksum in WEIGHTS:
        download(f'https://drive.usercontent.google.com/download?id={drive_id}&export=download&confirm=t',inputs/'weights'/filename,checksum)
    (inputs/'weights/SHA256SUMS').write_text(''.join(f'{checksum}  {filename}\n' for filename,_,checksum in WEIGHTS))
    for filename,checksum in IMAGES:
        download(f'https://sipi.usc.edu/database/misc/{filename}',inputs/'data'/filename,checksum)


def compare(actual: Path, expected: Path) -> None:
    """Require exact serialized values except Python and package version fields."""
    for filename in FILES:
        a=(actual/filename).read_bytes(); b=(expected/filename).read_bytes()
        if filename=='settings.json':
            left=json.loads(a); right=json.loads(b)
            for value in (left,right):
                value['environment'].pop('python')
                value['environment'].pop('packages')
            assert left==right,filename
        else:
            assert a==b,filename


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-inputs',type=Path,help='Offline root containing code/MBRS, weights, data')
    parser.add_argument('--output',type=Path,default=HERE/'reproduced')
    parser.add_argument('--check-results',type=Path,help='Compare an existing run with packaged results and exit')
    args=parser.parse_args()
    if args.check_results:
        compare(args.check_results,HERE/'results/run1')
        print('PASS: exact results, excluding environment version fields')
        return
    inputs=args.local_inputs.resolve() if args.local_inputs else args.output.resolve()/'inputs'
    if args.local_inputs is None:
        prepare(inputs)
    adaptations=json.loads((HERE/'api_adaptations.json').read_text())
    assert adaptations['changes']==[], 'This runtime has no required API patches.'
    environment={**os.environ,'MBRS_INPUTS':str(inputs),'PYTHONDONTWRITEBYTECODE':'1'}
    for name in ('run1','run2'):
        output=args.output.resolve()/name
        subprocess.run([sys.executable,'-B',str(HERE/'experiment/run_test.py'),'--output',str(output)],check=True,env=environment)
        compare(output,HERE/'results/run1')
    compare(args.output/'run1',args.output/'run2')
    print('PASS: both runs match each other and packaged numeric results exactly, excluding environment version fields')

if __name__=='__main__':
    main()
