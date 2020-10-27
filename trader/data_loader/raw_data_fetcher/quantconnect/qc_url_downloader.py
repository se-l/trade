import urllib.request
import shutil
import os
from concurrent.futures import ThreadPoolExecutor
from common.utils.util_func import create_dir

fn = 'FXCM_USDJPY_minute_quote'
base = r'C:\repos\trade2\trader\data_loader\qc_downloader\download_urls'
exchange, asset, res, qt = fn.split('_')
target_folder = r'C:\repos\quantconnect\Lean3\Data\forex\{}\{}\{}'.format(exchange.lower(), res, asset.lower())
create_dir(target_folder)


def file_exists(fn):
    try:
        with open(fn) as f:
            return True
    except IOError:
        return False


def del_file(fn):
    try:
        os.remove(fn)
    except FileNotFoundError:
        pass


def download(url):
    url = url.replace('\n', '')
    target_fn = os.path.join(target_folder, url.split('/')[-1]) + '_{}.zip'.format(qt)
    if file_exists(target_fn):
        return
    else:
        # Download the file from `url` and save it locally under `file_name`:
        print(f'Downloading {target_fn}')
        try:
            with urllib.request.urlopen(url) as response, open(target_fn, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except urllib.error.URLError as e:
            print(e)
            print(f'Failed Downloading {target_fn}')
            del_file(target_fn)


def main():
    with open(os.path.join(base, fn)) as f:
        urls = f. readlines()
    with ThreadPoolExecutor(max_workers=50) as executor:
        for _ in executor.map(download, urls):
            pass


if __name__ == '__main__':
    main()
