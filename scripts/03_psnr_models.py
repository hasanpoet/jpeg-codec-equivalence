from _common import arguments
if __name__=='__main__':
    arguments('psnr models')
    from src.analysis import analyze_pairs
    analyze_pairs()
    from src.models import run_psnr
    run_psnr()
