"""Parse actual framing bytes in the files retained for rate comparisons."""

from .config import WORKERS, crop_path


import ctypes


import gzip


import hashlib


import io


import json


import os


from concurrent.futures import ProcessPoolExecutor


from pathlib import Path


import imagecodecs


import numpy as np


import pandas as pd


from PIL import Image


from .analysis import OUT, CODECS, ACCEPTED, read, write_json


from .codecs import ProtocolFailure, canonical_pixels


_JXL = None


_DLL_DIRECTORY = None


def jxl_library():
    global _JXL, _DLL_DIRECTORY
    if _JXL is None:
        import importlib.metadata
        package=importlib.metadata.distribution('pillow-jxl-plugin')
        candidates=[Path(package.locate_file(f)) for f in package.files if 'jxl' in str(f).lower() and (str(f).endswith('.dll') or '.so' in str(f) or str(f).endswith('.dylib')) and 'threads' not in str(f)]
        libraries=[p for p in candidates if p.name.startswith('libjxl-') or p.name.startswith('libjxl.')]
        if not libraries:raise ProtocolFailure('Pinned pillow-jxl-plugin native libjxl was not found')
        library=libraries[0]
        if os.name=='nt':_DLL_DIRECTORY=os.add_dll_directory(str(library.parent))
        _JXL=ctypes.CDLL(str(library))
        signatures = {
            'JxlDecoderCreate': ([ctypes.c_void_p], ctypes.c_void_p),
            'JxlDecoderDestroy': ([ctypes.c_void_p], None),
            'JxlDecoderSubscribeEvents': ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
            'JxlDecoderSetInput': ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            'JxlDecoderProcessInput': ([ctypes.c_void_p], ctypes.c_int),
            'JxlDecoderReleaseInput': ([ctypes.c_void_p], ctypes.c_size_t),
            'JxlDecoderVersion': ([], ctypes.c_uint32),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(_JXL, name)
            fn.argtypes, fn.restype = args, result
    return _JXL


def jpeg_header(data):
    if data[:2] != b'\xff\xd8' or data[-2:] != b'\xff\xd9':
        raise ProtocolFailure('Corrupted baseline JPEG framing')
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 255:
            raise ProtocolFailure('Invalid JPEG marker')
        marker = data[pos + 1]
        length = int.from_bytes(data[pos+2:pos+4], 'big')
        if length < 2 or pos + 2 + length > len(data):
            raise ProtocolFailure('Corrupted JPEG segment')
        pos += 2 + length
        if marker == 0xDA:
            return pos + 2  # Prefix through SOS, plus final EOI marker.
    raise ProtocolFailure('JPEG has no scan')


def jp2_header(data):
    pos, codestream, overhead = 0, None, 0
    while pos < len(data):
        length = int.from_bytes(data[pos:pos+4], 'big')
        kind, header = data[pos+4:pos+8], 8
        if length == 1:
            length, header = int.from_bytes(data[pos+8:pos+16], 'big'), 16
        elif length == 0:
            length = len(data) - pos
        if length < header or pos + length > len(data):
            raise ProtocolFailure('Corrupted JP2 box')
        if kind == b'jp2c':
            if codestream is not None:
                raise ProtocolFailure('Unexpected multiple JP2 codestreams')
            codestream = data[pos+header:pos+length]
            overhead += header
        else:
            overhead += length
        pos += length
    if codestream is None or codestream[:2] != b'\xffO' or codestream[-2:] != b'\xff\xd9':
        raise ProtocolFailure('Corrupted JP2 codestream')
    pos, tiles = 2, 0
    while pos + 2 <= len(codestream):
        marker = codestream[pos:pos+2]
        if marker == b'\xff\x90':
            tiles += 1
        if marker == b'\xff\x93':
            if tiles != 1:
                raise ProtocolFailure('Expected one JPEG 2000 tile')
            return overhead + pos + 2 + 2  # SOD and EOC framing markers.
        if marker[0] != 255:
            raise ProtocolFailure('Invalid JPEG 2000 marker')
        length = int.from_bytes(codestream[pos+2:pos+4], 'big')
        if length < 2:
            raise ProtocolFailure('Invalid JPEG 2000 marker length')
        pos += 2 + length
    raise ProtocolFailure('JPEG 2000 has no tile data')


def jxl_header(data):
    # The public decoder API defines ReleaseInput at JXL_DEC_FRAME as the
    # boundary after the frame header including its table of contents.
    # https://libjxl.readthedocs.io/en/latest/api_decoder.html
    lib = jxl_library()
    decoder = lib.JxlDecoderCreate(None)
    if not decoder:
        raise ProtocolFailure('JPEG XL header decoder allocation failed')
    try:
        buffer = ctypes.create_string_buffer(data)
        if lib.JxlDecoderSubscribeEvents(decoder, 0x400) != 0 or lib.JxlDecoderSetInput(decoder, buffer, len(data)) != 0:
            raise ProtocolFailure('JPEG XL header decoder setup failed')
        if lib.JxlDecoderProcessInput(decoder) != 0x400:
            raise ProtocolFailure('JPEG XL frame header could not be parsed')
        return len(data) - lib.JxlDecoderReleaseInput(decoder)
    finally:
        lib.JxlDecoderDestroy(decoder)


def encoded_file(a, codec, parameter):
    a = canonical_pixels(a)
    if codec == 'jpegxl':
        data = imagecodecs.jpegxl_encode(a, distance=float(parameter), effort=7, numthreads=1)
        recon = imagecodecs.jpegxl_decode(data, numthreads=1)
    else:
        f = io.BytesIO()
        image = Image.fromarray(a)
        if codec == 'jpeg':
            image.save(f, format='JPEG', quality=int(parameter), progressive=False, subsampling=2, optimize=False)
        else:
            image.save(f, format='JPEG2000', quality_mode='rates', quality_layers=[float(parameter)],
                       irreversible=True, mct=1 if a.ndim == 3 else 0, tile_size=image.size)
        data = f.getvalue()
        recon = np.array(Image.open(io.BytesIO(data)))
    if recon.shape != a.shape or recon.dtype != a.dtype:
        raise ProtocolFailure('Header-pass codec round-trip mismatch')
    return data


def header_image(task):
    record, pairs, fingerprint = task
    dest = OUT / 'checkpoints/headers' / record['domain'] / (record['image_id'] + '.json.gz')
    if dest.exists():
        with gzip.open(dest, 'rt') as f:
            checkpoint = json.load(f)
        if checkpoint['fingerprint'] != fingerprint:
            raise ProtocolFailure('Header checkpoint mismatch')
        return str(dest)
    a = np.array(Image.open(crop_path(record)))
    if hashlib.sha256(a.tobytes()).hexdigest() != record['sha256']:
        raise ProtocolFailure('Header-pass pixel hash mismatch')
    cache, rows = {}, []
    for pair in pairs:
        targets = [('jpeg', 'target', pair['q'], pair['jpeg_bytes'])]
        for codec in CODECS:
            for metric in ['psnr', 'ssim']:
                p = codec + '_' + metric + '_'
                if pair[p + 'status'] in ACCEPTED:
                    targets.append((codec, metric, pair[p + 'parameter'], pair[p + 'encoded_bytes']))
        for codec, metric, parameter, size in targets:
            key = codec, parameter
            if key not in cache:
                data = encoded_file(a, codec, parameter)
                header = dict(jpeg=jpeg_header, jpeg2000=jp2_header, jpegxl=jxl_header)[codec](data)
                cache[key] = len(data), header
            actual, header = cache[key]
            if actual != size or not 0 < header <= size:
                raise ProtocolFailure('Header pass disagrees with retained file size or has invalid header length')
            rows.append(dict(domain=record['domain'], image_id=record['image_id'], q=pair['q'],
                             codec=codec, metric=metric, parameter=parameter, encoded_bytes=actual,
                             header_bytes=header, header_pct=100 * header / actual))
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.tmp')
    with gzip.open(temp, 'wt') as f:
        json.dump(dict(fingerprint=fingerprint, rows=rows), f)
    temp.replace(dest)
    return str(dest)


def run_headers():
    from .environment import verify
    verify()
    manifest, pairs = read('manifest.csv'), read('pairs.csv')
    grouped = {(d, i): f.to_dict('records') for (d, i), f in pairs.groupby(['domain', 'image_id'])}
    fingerprint = hashlib.sha256((OUT / 'pairs.csv').read_bytes()).hexdigest()
    tasks = [(r, grouped[r['domain'], r['image_id']], fingerprint) for r in manifest.to_dict('records')]
    frames = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i, path in enumerate(pool.map(header_image, tasks, chunksize=1)):
            with gzip.open(path, 'rt') as f:
                frames.append(pd.DataFrame(json.load(f)['rows']))
            if (i+1) % 25 == 0:
                print(f'Parsed headers: {i+1}/{len(tasks)} images', flush=True)
    headers = pd.concat(frames, ignore_index=True)
    headers.to_csv(OUT / 'header_measurements.csv', index=False)
    maximum = headers.loc[headers.header_pct.idxmax()].to_dict()
    write_json('header_check.json', dict(status='passed', files=len(headers), maximum=maximum,
        scope='JPEG targets and retained matched/matched_scan equivalence files for PSNR and SSIM; excludes discarded search trials and model-prediction encodes',
        definition='Container, metadata and codestream/frame headers before compressed image data; JPEG EOI and JPEG 2000 EOC included; JXL frame ToC included',
        jpegxl_parser_version=int(jxl_library().JxlDecoderVersion()),
        jpegxl_parser_reference='https://libjxl.readthedocs.io/en/latest/api_decoder.html#c.JXL_DEC_FRAME'))
    return float(maximum['header_pct'])
