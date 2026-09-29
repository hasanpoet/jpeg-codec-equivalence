from _common import arguments
if __name__=='__main__':
    arguments('ssim models')
    from src.robustness import run_ssim
    run_ssim()
