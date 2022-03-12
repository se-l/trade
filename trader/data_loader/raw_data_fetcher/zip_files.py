import os
import gzip
import shutil

from common.paths import Paths
from common.modules.logger import logger

if __name__ == '__main__':
    for _, dirs_, fns in os.walk(Paths.bitfinex_tick):
        # for dir_ in dirs_:
        for fn in fns:
            if not fn.endswith('.gz'):
                with open(os.path.join(_, fn), 'rb') as f_in:
                    logger.info(f'zipping {f_in}...')
                    with gzip.open(os.path.join(_, fn + '.gz'), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(os.path.join(_, fn))
    logger.info(f'Done.')
