import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache'


@dataclass(frozen=True)
class Download:
    url: str
    path: Path
    size: Optional[int]
    sha256: str


class VerificationError(RuntimeError):
    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__('Checksum or size mismatch: ' + str(path.relative_to(ROOT)))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify(download: Download, path: Path) -> None:
    if ((download.size is not None and path.stat().st_size != download.size)
            or digest(path) != download.sha256):
        raise VerificationError(path)


def fetch(download: Download) -> None:
    download.path.parent.mkdir(parents=True, exist_ok=True)
    if download.path.exists():
        verify(download, download.path)
        return
    partial = download.path.with_name(download.path.name + '.part')
    subprocess.run(['curl', '--fail', '--location', '--retry', '2',
                    '--connect-timeout', '20', '--max-time', '1200',
                    '--output', str(partial), download.url], check=True)
    verify(download, partial)
    partial.replace(download.path)
