import argparse
import os
import string
import random
import zipfile

import gdown

parser = argparse.ArgumentParser(description='Download and unpack NCAA Dataset')
parser.add_argument('output', help="Output directory path for dataset.", type=str)

args = parser.parse_args()

if __name__ == "__main__":
    temp_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=32)) + ".zip"
    base_url = "https://drive.google.com/file/d/15CuwcvCg1OAQMDtsVKyoD-mR-bMPvubu/view?usp=sharing"
    direct_url = "https://drive.google.com/uc?export=download&id=" + base_url.split("/")[-2]

    try:
        gdown.download(direct_url, temp_filename, quiet=False)
        print("Unpacking downloaded zip...")
        with zipfile.ZipFile(temp_filename, 'r') as zip_ref:
            zip_ref.extractall(args.output)
    finally:
        os.remove(temp_filename)
