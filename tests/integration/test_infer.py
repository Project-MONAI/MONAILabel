import unittest

import requests

from . import SERVER_URI


class EndPointInfer(unittest.TestCase):
    def test_segmentation(self):
        model = "segmentation_left_atrium"
        image = "la_004.nii.gz"

        response = requests.post(f"{SERVER_URI}/infer/{model}?image={image}")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
