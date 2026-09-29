from _common import arguments
if __name__=='__main__':
    arguments('Reference quantization-table sensitivity and PSNR ratio slope signs')
    from src.reference_robustness import main
    main()
