"""Strict codec round trips and fidelity measurements for native-resolution crops."""


import io


import time


import imagecodecs


import numpy as np


from PIL import Image


from skimage.metrics import structural_similarity


BOUNDS = {'jpeg2000': (1.5, 1000.), 'jpegxl': (.05, 25.)}


QUALITIES = list(range(10, 100, 10))


class ProtocolFailure(RuntimeError):
    pass


def canonical_pixels(a):
    if a.ndim == 3 and a.shape[2] == 3 and np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2]):
        a = a[..., 0]
    if a.dtype != np.uint8 or not (a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 3)):
        raise ProtocolFailure(f'Unsupported input dtype/shape: {a.dtype}/{a.shape}')
    return np.ascontiguousarray(a)


def luma(a):
    return a.astype(float) if a.ndim == 2 else np.asarray(Image.fromarray(a).convert('YCbCr'))[..., 0].astype(float)


def encode(a, codec, parameter):
    """Always compare decoded shape AND dtype, without coercing the decoded image."""
    a = canonical_pixels(a)
    start = time.perf_counter()
    if codec == 'jpegxl':
        data = imagecodecs.jpegxl_encode(a, distance=float(parameter), effort=7, numthreads=1)
        recon = imagecodecs.jpegxl_decode(data, numthreads=1)
    else:
        f = io.BytesIO()
        im = Image.fromarray(a)
        if codec == 'jpeg':
            im.save(f, format='JPEG', quality=int(parameter), progressive=False, subsampling=2, optimize=False)
        elif codec == 'jpeg2000':
            im.save(f, format='JPEG2000', quality_mode='rates', quality_layers=[float(parameter)],
                    irreversible=True, mct=1 if a.ndim == 3 else 0, tile_size=im.size)
        else:
            raise ValueError(codec)
        data = f.getvalue()
        recon = np.array(Image.open(io.BytesIO(data)))
    if recon.shape != a.shape or recon.dtype != a.dtype:
        raise ProtocolFailure(f'{codec} round-trip mismatch: {a.shape}/{a.dtype} -> {recon.shape}/{recon.dtype}')
    mse = np.mean((a.astype(float)-recon.astype(float))**2)
    p = float('inf') if mse == 0 else float(10*np.log10(255**2/mse))
    s = float(structural_similarity(luma(a), luma(recon), data_range=255))
    return dict(parameter=float(parameter), psnr=p, ssim=s, encoded_bytes=len(data),
                bpp=8*len(data)/a.shape[0]/a.shape[1], seconds=time.perf_counter()-start)
