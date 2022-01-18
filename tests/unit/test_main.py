import argparse
import os
import shutil
import unittest

from monailabel.main import Main


class MyTestCase(unittest.TestCase):
    client = None
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    data_dir = os.path.join(base_dir, "tests", "data")

    app_dir = os.path.join(base_dir, "sample-apps", "deepedit")
    studies = os.path.join(data_dir, "dataset", "local", "heart")

    def test_run(self):
        try:
            Main().run()
        except BaseException:
            pass

    def test_start_server(self):
        args = argparse.Namespace(
            app=self.app_dir,
            studies=self.studies,
            username=None,
            password=None,
            wado_prefix="",
            qido_prefix="",
            stow_prefix="",
            debug=False,
            dryrun=True,
            conf={},
            host="0.0.0.0",
            port=8000,
            log_config=None,
        )
        Main().action_start_server(args)

    def test_apps(self):
        args = argparse.Namespace(download=False)
        Main().action_apps(args)

    def test_apps_download(self):
        output = os.path.join(self.data_dir, "downloaded_app")
        args = argparse.Namespace(download=True, name="segmentation", output=output)
        Main().action_apps(args)
        assert os.path.isdir(output)
        shutil.rmtree(output, ignore_errors=True)

    def test_datasets(self):
        args = argparse.Namespace(download=False)
        Main().action_datasets(args)

    def test_datasets_download(self):
        output = os.path.join(self.data_dir, "downloaded_dataset")
        args = argparse.Namespace(download=True, name="Task04_Hippocampus", output=output)
        Main().action_datasets(args)
        assert os.path.isdir(output)
        shutil.rmtree(output, ignore_errors=True)

    def test_plugins(self):
        args = argparse.Namespace(download=False)
        Main().action_plugins(args)

    def test_plugins_download(self):
        output = os.path.join(self.data_dir, "downloaded_plugins")
        args = argparse.Namespace(download=True, name="slicer", output=output)
        Main().action_plugins(args)
        assert os.path.isdir(output)
        shutil.rmtree(output, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
