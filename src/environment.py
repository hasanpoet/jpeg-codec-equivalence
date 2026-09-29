"""Reject native codec changes before generating incompatible measurements."""
import json
import imagecodecs
from PIL import Image,features
from .config import ROOT
from .codecs import ProtocolFailure

def verify():
    actual=dict(Pillow=Image.__version__,Libjpeg=features.version_codec('jpg'),LibjpegTurbo=features.version_feature('libjpeg_turbo'),Openjpeg=features.version_codec('jpg_2000'),Libjxl=imagecodecs.jpegxl_version(),Imagecodecs=imagecodecs.__version__)
    expected=json.loads((ROOT/'results/encoder_versions.json').read_text())
    if actual!=expected:raise ProtocolFailure('Native codec versions differ from results/encoder_versions.json')
    return actual
