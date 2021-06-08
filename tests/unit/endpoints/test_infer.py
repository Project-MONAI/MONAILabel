import json
import unittest

from .context import BasicEndpointTestSuite


class EndPointInfer(BasicEndpointTestSuite):
    @unittest.skip("GPU Needed")
    def test_segmentation(self):
        model = "heart"
        image = "la_003.nii.gz"

        response = self.client.post(f"/infer/{model}?image={image}")
        assert response.status_code == 200

    @unittest.skip("GPU Needed")
    def test_deepedit(self):
        model = "deepedit"
        image = "la_003.nii.gz"
        params = {"foreground": [[153, 175, 60]], "background": []}

        response = self.client.post(f"/infer/{model}?image={image}", data={"params": json.dumps(params)})
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
