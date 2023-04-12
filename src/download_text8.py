import urllib.request
import os


def download_text8(url, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    target_file = os.path.join(target_folder, 'text8.zip')

    if not os.path.isfile(target_file):
        print("Downloading text8 dataset...")
        urllib.request.urlretrieve(url, target_file)
        print("Download complete.")


if __name__ == "__main__":
    text8_url = "http://mattmahoney.net/dc/text8.zip"
    target_folder = "text8_dataset"

    download_text8(text8_url, target_folder)
