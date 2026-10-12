"""Lossless bounded files for Git archival; never deletes numerical source data."""
import hashlib
import json
from pathlib import Path
import tempfile

LIMIT = 48 * 1024**2
BLOCK = 1024**2


def split(source, directory, expected_sha256, limit=LIMIT):
    source, directory = Path(source), Path(directory)
    if type(limit) is not int or not 0 < limit <= LIMIT:
        raise ValueError('Positive integer chunk size at most48MiB required')
    if len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('Registered SHA256 required')
    directory.mkdir(exist_ok=False)
    digest = hashlib.sha256(); parts = []; size = 0
    with source.open('rb') as stream:
        while data := stream.read(limit):
            name = f'part_{len(parts):06d}'
            with (directory / name).open('xb') as output:
                output.write(data)
            parts.append(dict(name=name, offset=size, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            digest.update(data); size += len(data)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('Source differs from its registered hash; partial chunks retained')
    manifest = dict(version=1, source_name=source.name, bytes=size, sha256=expected_sha256,
                    limit_bytes=limit, parts=parts, source_deleted=False)
    with (directory / 'manifest.json').open('x') as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write('\n')
    return manifest


def verify(manifest_path):
    manifest_path = Path(manifest_path); manifest = json.loads(manifest_path.read_text())
    assert manifest['version'] == 1 and type(manifest['bytes']) is int and manifest['bytes'] >= 0
    assert type(manifest['limit_bytes']) is int and 0 < manifest['limit_bytes'] <= LIMIT
    digest = hashlib.sha256(); offset = 0
    for index, part in enumerate(manifest['parts']):
        assert part['name'] == f'part_{index:06d}' and type(part['offset']) is int and part['offset'] == offset
        assert type(part['bytes']) is int and 0 < part['bytes'] <= manifest['limit_bytes']
        path = manifest_path.parent / part['name']
        assert path.is_file() and not path.is_symlink() and path.stat().st_size == part['bytes']
        local = hashlib.sha256()
        with path.open('rb') as stream:
            while data := stream.read(BLOCK):
                local.update(data); digest.update(data)
        assert local.hexdigest() == part['sha256'], 'Chunk changed'
        offset += part['bytes']
    assert offset == manifest['bytes'] and digest.hexdigest() == manifest['sha256'], 'Incomplete archive'
    return manifest


def restore(manifest_path, destination):
    manifest_path, destination = Path(manifest_path), Path(destination)
    manifest = verify(manifest_path); digest = hashlib.sha256()
    with destination.open('xb') as output:
        for part in manifest['parts']:
            with (manifest_path.parent / part['name']).open('rb') as stream:
                while data := stream.read(BLOCK):
                    output.write(data); digest.update(data)
    assert digest.hexdigest() == manifest['sha256'], 'Archive changed during restore; partial output retained'
    return manifest


def self_check():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); data = bytes(range(256))*4; source = root/'source.npz'; source.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        result = split(source, root/'chunks', sha, 31)
        assert max(p['bytes'] for p in result['parts']) <= 31 and source.read_bytes() == data
        restore(root/'chunks/manifest.json', root/'restored.npz')
        assert (root/'restored.npz').read_bytes() == data
        for limit in (True, 0, LIMIT+1):
            try: split(source, root/'invalid', sha, limit)
            except ValueError: pass
            else: raise AssertionError('Invalid limit accepted')
        try: restore(root/'chunks/manifest.json', source)
        except FileExistsError: pass
        else: raise AssertionError('Existing numerical source overwritten')
        (root/'chunks/part_000000').write_bytes(bytes(31))
        try: restore(root/'chunks/manifest.json', root/'bad.npz')
        except AssertionError: pass
        else: raise AssertionError('Corrupted archive accepted')
        assert not (root/'bad.npz').exists() and source.read_bytes() == data
        (root/'chunks/part_000000').unlink()
        try: verify(root/'chunks/manifest.json')
        except AssertionError: pass
        else: raise AssertionError('Missing chunk accepted')
        empty = root/'empty.npz'; empty.write_bytes(b'')
        split(empty, root/'emptychunks', hashlib.sha256(b'').hexdigest())
        restore(root/'emptychunks/manifest.json', root/'emptyrestored.npz')
        assert not (root/'emptyrestored.npz').read_bytes()
    print('PASS381 bounded lossless chunks,whole-byte restore,corruption/missingchunk/invalidlimit/overwrite rejection;no source deletion/physics/FD/NN')


if __name__ == '__main__': self_check()
