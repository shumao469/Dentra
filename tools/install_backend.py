"""One-time isolated local deployment; downloads public weights, never patient images."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if sys.platform != 'linux':
        raise RuntimeError('Run this deployment helper in WSL/Linux, not in Slicer Python')
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-pip', action='store_true')
    parser.add_argument('--provider', choices=['dental', 'toothseg'], default='dental')
    args = parser.parse_args()
    config = json.loads((ROOT / 'config' / 'local_backend.json').read_text())
    is_toothseg = args.provider == 'toothseg'
    manifest = json.loads((ROOT / 'config' / ('toothseg_manifest.json' if is_toothseg else 'model_manifest.json')).read_text())
    cache = Path(config['toothseg_model_root' if is_toothseg else 'model_root']).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    archive_name = 'ToothSeg.zip' if is_toothseg else 'Dataset112_DentalSegmentator_v100.zip'
    record = '14893540' if is_toothseg else '10829675'
    final = cache / archive_name
    partial = final.with_suffix('.zip.partial')
    expected = manifest['archive_bytes']
    if final.exists() and (final.stat().st_size != expected or digest(final) != manifest['archive_sha256']):
        raise RuntimeError('Existing archive checksum failed; preserve it for inspection and choose a clean cache')
    for attempt in range(15):
        if final.exists():
            break
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise RuntimeError('Partial archive exceeds the published size')
        if offset < expected:
            request = urllib.request.Request('https://zenodo.org/api/records/' + record + '/files/' + archive_name + '/content',
                headers={'Range': f'bytes={offset}-{expected-1}', 'User-Agent': 'CBCT-local-draft/0.2'})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                        raise RuntimeError('Server did not honor download resume')
                    with partial.open('ab' if offset else 'wb') as target:
                        while block := response.read(4 * 1024 * 1024):
                            target.write(block)
                print('Download bytes', partial.stat().st_size, '/', expected, flush=True)
            except Exception as exc:
                print('Download interrupted; partial retained:', str(exc), flush=True)
                time.sleep(2)
                continue
        if partial.stat().st_size == expected:
            if digest(partial, 'md5') != manifest['archive_md5'] or digest(partial) != manifest['archive_sha256']:
                raise RuntimeError('Downloaded model hash failed; no weights will be loaded')
            partial.rename(final)
    if not final.exists():
        raise RuntimeError('Download incomplete; partial retained for a later retry')
    with zipfile.ZipFile(final) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive CRC failed')
        for relative, expected_hash in manifest['files_sha256'].items():
            target = (cache / relative).resolve()
            if not target.is_relative_to(cache):
                raise RuntimeError('Model extraction path rejected')
            if target.exists():
                if digest(target) != expected_hash:
                    raise RuntimeError('Existing model file changed: ' + relative)
                continue
            data = archive.read(relative)
            if hashlib.sha256(data).hexdigest() != expected_hash:
                raise RuntimeError('Model member checksum failed')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    python = Path(config['python'])
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(python.parent.parent)], check=True)
    if not args.skip_pip:
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'config' / 'requirements-backend.txt')], check=True)
    subprocess.run([str(python), str(ROOT / 'backend' / ('toothseg_infer.py' if is_toothseg else 'infer.py')), '--preflight'], check=True)


if __name__ == '__main__':
    main()
