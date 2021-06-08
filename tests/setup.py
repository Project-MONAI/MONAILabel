import os

from monai.apps import download_url, extractall

TEST_DATA = os.path.join(os.path.realpath(os.path.dirname(__file__)), "data")


def run_main():
    dataset_file = os.path.join(TEST_DATA, "dataset.zip")
    dataset_url = "https://www.dropbox.com/s/h1euuclz4c3ekxt/test_dataset.zip?dl=1"
    if not os.path.exists(os.path.join(TEST_DATA, "dataset")):
        if not os.path.exists(dataset_file):
            download_url(url=dataset_url, filepath=dataset_file)
        extractall(filepath=dataset_file, output_dir=TEST_DATA)


if __name__ == "__main__":
    run_main()
