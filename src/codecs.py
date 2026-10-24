"""Strict codec round trips and fidelity measurements for native-resolution crops."""


import io


import time


from dataclasses import dataclass


import imagecodecs


import numpy as np


from PIL import Image


from skimage.metrics import structural_similarity


BOUNDS = {'jpeg2000': (1.5, 1000.), 'jpegxl': (.05, 25.)}


QUALITIES = list(range(10, 100, 10))


class ProtocolFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class EncoderConfig:
    """Material settings; defaults preserve the published protocol."""
    effort: int = 7
    subsampling: int = 2
    irreversible: bool = True
    color_transform: bool = True


def canonical_pixels(a):
    if a.ndim == 3 and a.shape[2] == 3 and np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2]):
        a = a[..., 0]
    if a.dtype != np.uint8 or not (a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 3)):
        raise ProtocolFailure(f'Unsupported input dtype/shape: {a.dtype}/{a.shape}')
    return np.ascontiguousarray(a)


def luma(a):
    return a.astype(float) if a.ndim == 2 else np.asarray(Image.fromarray(a).convert('YCbCr'))[..., 0].astype(float)


def encode(a, codec, parameter, config=None, return_data=False):
    """Always compare decoded shape AND dtype, without coercing the decoded image."""
    a = canonical_pixels(a)
    config = config or EncoderConfig()
    start = time.perf_counter()
    if codec == 'jpegxl':
        data = imagecodecs.jpegxl_encode(a, distance=float(parameter), effort=config.effort, numthreads=1)
        recon = imagecodecs.jpegxl_decode(data, numthreads=1)
    else:
        f = io.BytesIO()
        im = Image.fromarray(a)
        if codec == 'jpeg':
            im.save(f, format='JPEG', quality=int(parameter), progressive=False, subsampling=config.subsampling, optimize=False)
        elif codec == 'jpeg2000':
            im.save(f, format='JPEG2000', quality_mode='rates', quality_layers=[float(parameter)],
                    irreversible=config.irreversible, mct=int(config.color_transform and a.ndim == 3), tile_size=im.size)
        else:
            raise ValueError(codec)
        data = f.getvalue()
        recon = np.array(Image.open(io.BytesIO(data)))
    if recon.shape != a.shape or recon.dtype != a.dtype:
        raise ProtocolFailure(f'{codec} round-trip mismatch: {a.shape}/{a.dtype} -> {recon.shape}/{recon.dtype}')
    mse = np.mean((a.astype(float)-recon.astype(float))**2)
    p = float('inf') if mse == 0 else float(10*np.log10(255**2/mse))
    s = float(structural_similarity(luma(a), luma(recon), data_range=255))
    result = dict(parameter=float(parameter), psnr=p, ssim=s, encoded_bytes=len(data),
                  bpp=8*len(data)/a.shape[0]/a.shape[1], seconds=time.perf_counter()-start)
    return (result, data) if return_data else result
