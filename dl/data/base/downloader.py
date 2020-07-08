import os
import pycurl
import tarfile, zipfile
import glob, shutil
import logging

class _Downloader:
    def __init__(self, url, compress_ext='tar'):
        self.url = url

        _compress_exts = ['tar', 'zip']
        if not compress_ext in _compress_exts:
            raise ValueError("Invalid argument, select proper extension from {}, but got {}".format(_compress_exts, compress_ext))
        self.compress_ext = compress_ext

    def run(self, out_base_dir, dirname, remove_comp_file=True):
        out_dir = os.path.join(out_base_dir, dirname)

        if len(glob.glob(os.path.join(out_dir, '*'))) > 0:
            logging.warning('dataset may be already downloaded. If you haven\'t done yet, remove \"{}\" directory'.format(out_base_dir))
            return

        curl = pycurl.Curl()
        curl.setopt(pycurl.URL, self.url)
        # allow redirect
        curl.setopt(pycurl.FOLLOWLOCATION, True)
        # show progress
        curl.setopt(pycurl.NOPROGRESS, False)

        os.makedirs(out_dir, exist_ok=True)

        dstpath = os.path.join(out_base_dir, '{}.{}'.format(dirname, self.compress_ext))

        with open(dstpath, 'wb') as f:
            curl.setopt(pycurl.WRITEFUNCTION, f.write)
            curl.perform()

        curl.close()


        # extract
        if self.compress_ext == 'tar':
            with tarfile.open(dstpath) as tar_f:
                tar_f.extractall(out_dir)
        elif self.compress_ext == 'zip':
            with zipfile.ZipFile(dstpath, 'r') as zip_f:
                zip_f.extractall(out_dir)
        else:
            assert False, "Bug occurred"

        if remove_comp_file:
            # remove tmp.*
            os.remove(dstpath)