import unittest

import torch

from .context import BasicEndpointTestSuite


class TestEndPointLogs(BasicEndpointTestSuite):
    def test_logs(self):
        response = self.client.get("/logs/")
        assert response.status_code == 200

    def test_logs_all(self):
        response = self.client.get("/logs/?lines=0")
        assert response.status_code == 200

    def test_logs_gpu(self):
        if not torch.cuda.is_available():
            return

        response = self.client.get("/logs/gpu")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
