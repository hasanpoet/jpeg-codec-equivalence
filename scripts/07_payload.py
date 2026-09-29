from _common import arguments
if __name__=='__main__':
    arguments('Compute payload-only rates from saved per-file header measurements')
    from src.payload import run_payload
    run_payload()
