import json
import unittest

import torch

from .context import BasicEndpointTestSuite


class EndPointInfer(BasicEndpointTestSuite):
    def test_segmentation(self):
        model = "heart"
        image = "la_003.nii.gz"
        response = self.client.post(f"/infer/{model}?image={image}")
        if torch.cuda.is_available():
            assert response.status_code == 200

    def test_deepedit(self):
        model = "deepedit"
        image = "la_003.nii.gz"
        params = {"foreground": [[153, 175, 60]], "background": []}

        response = self.client.post(f"/infer/{model}?image={image}", data={"params": json.dumps(params)})
        if torch.cuda.is_available():
            assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
