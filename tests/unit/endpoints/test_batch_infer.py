import unittest

import torch

from .context import BasicEndpointTestSuite


class EndPointScoring(BasicEndpointTestSuite):
    def test_batch_infer(self):
        if not torch.cuda.is_available():
            return

        model = "heart"
        images = "labeled"
        response = self.client.post(f"/batch/infer/{model}?images={images}&run_sync=true")
        assert response.status_code == 200

    def test_status(self):
        self.client.get("/batch/infer/")

    def test_stop(self):
        self.client.delete("/batch/infer/")


if __name__ == "__main__":
    unittest.main()
