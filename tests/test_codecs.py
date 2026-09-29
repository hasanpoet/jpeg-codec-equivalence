import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from src.codecs import ProtocolFailure,encode

class CodecTests(unittest.TestCase):
    def test_shape_and_dtype_are_not_silently_coerced(self):
        a=np.zeros((16,16),dtype=np.uint8)
        for bad in [np.zeros((16,16,3),dtype=np.uint8),a.astype(np.uint16)]:
            fake=SimpleNamespace(jpegxl_encode=lambda *a,**k:b'encoded',jpegxl_decode=lambda *a,**k:bad)
            with patch('src.codecs.imagecodecs',fake):
                with self.assertRaises(ProtocolFailure):encode(a,'jpegxl',1.)


    def test_jpeg2000_overrides_default_wavelet_and_color_transform(self):
        from PIL import Image
        real_save=Image.Image.save
        for shape,mct in [((32,32),0),((32,32,3),1)]:
            array=np.random.default_rng(2).integers(0,256,shape,dtype=np.uint8)
            seen=[]
            def save(image,stream,format=None,**kwargs):
                seen.append(kwargs)
                return real_save(image,stream,format=format,**kwargs)
            with patch.object(Image.Image,'save',save):encode(array,'jpeg2000',4.)
            self.assertEqual(seen[0]['mct'],mct)
            self.assertTrue(seen[0]['irreversible'])
            self.assertEqual(seen[0]['quality_layers'],[4.])
            self.assertEqual(seen[0]['tile_size'],(32,32))
