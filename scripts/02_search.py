"""Run all nine JPEG levels and PSNR/SSIM searches for both target codecs."""
from _common import arguments
if __name__=='__main__':
    arguments(__doc__)
    from src.search import run
    run()
