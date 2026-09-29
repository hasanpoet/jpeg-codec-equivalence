from _common import arguments
if __name__=='__main__':
    arguments('regenerate')
    from src.reporting import regenerate
    regenerate()
