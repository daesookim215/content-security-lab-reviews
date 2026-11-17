import json
from pathlib import PurePosixPath
import shutil
import tarfile
from download_file import ROOT, CACHE, Download, VerificationError, digest, fetch


def main() -> None:
    manifest = json.loads((ROOT / 'code/source_manifest.json').read_text())
    archive = CACHE / 'official-source.tar.gz'
    fetch(Download(manifest['url'], archive, manifest['bytes'], manifest['sha256']))
    target = CACHE / 'official-source'
    target.mkdir(parents=True, exist_ok=True)
    found = set()
    with tarfile.open(str(archive), 'r:gz') as source:
        for entry in source.getmembers():
            path = PurePosixPath(entry.name)
            if (path.is_absolute() or '..' in path.parts or '\\' in entry.name
                    or not path.parts or path.parts[0] != 'tancik-StegaStamp-c984446'
                    or not (entry.isdir() or entry.isfile())):
                raise VerificationError(archive)
            relative = str(PurePosixPath(*path.parts[1:]))
            if relative not in manifest['files']:
                continue
            if not entry.isfile() or relative in found:
                raise VerificationError(archive)
            found.add(relative)
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(entry)
            if stream is None:
                raise VerificationError(archive)
            with stream, output.open('wb') as destination:
                shutil.copyfileobj(stream, destination)
            if digest(output) != manifest['files'][relative]:
                raise VerificationError(output)
    if found != set(manifest['files']):
        raise VerificationError(archive)
    print('PASS: pinned official archive and source hashes')


if __name__ == '__main__':
    main()
