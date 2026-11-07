"""Fetch the two USC-SIPI inputs and require their recorded SHA-256 hashes."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.error import URLError
from urllib.request import urlopen

ROOT: Final = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class Source:
    name: str
    filename: str
    digest: str


SOURCES: Final = (
    Source("airplane", "4.2.05.tiff", "ae9eba12b80df955e47585bf4a16332ea0dadc753668d3457c509564934dbd40"),
    Source("boat", "boat.512.tiff", "6bc7319b1e3b44ce0b2a571ac39a28bac94437d812057d1afb2cf78ea3a9a35f"),
)


def verify_file(source: Source) -> Path:
    path = ROOT / "images" / source.filename
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != source.digest:
        raise SystemExit(f"FAIL checksum {source.filename}: expected {source.digest}, got {actual}")
    return path


def main() -> None:
    (ROOT / "images").mkdir(exist_ok=True)
    for source in SOURCES:
        destination = ROOT / "images" / source.filename
        if not destination.exists():
            url = "https://sipi.usc.edu/database/misc/" + source.filename
            try:
                with urlopen(url, timeout=90) as response:
                    data = response.read()
            except (URLError, TimeoutError) as error:
                raise SystemExit(f"FAIL download {source.filename}: {error}") from error
            actual = sha256(data).hexdigest()
            if actual != source.digest:
                raise SystemExit(f"FAIL checksum {source.filename}: expected {source.digest}, got {actual}")
            destination.write_bytes(data)
        verify_file(source)
        print(f"PASS SHA-256 {source.filename}: {source.digest}")


if __name__ == "__main__":
    main()
