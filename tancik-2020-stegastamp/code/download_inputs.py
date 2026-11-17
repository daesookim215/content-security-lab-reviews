import json
from download_file import ROOT, CACHE, Download, fetch


def main() -> None:
    for row in json.loads((ROOT / 'code/input_manifest.json').read_text()):
        fetch(Download(row['url'], CACHE / 'inputs' / row['filename'], None, row['sha256']))
    print('PASS: both USC-SIPI input hashes')


if __name__ == '__main__':
    main()
