import unittest

from .context import BasicEndpointTestSuite


class EndPointScoring(BasicEndpointTestSuite):
    def test_dice(self):
        response = self.client.post("/scoring/dice?force_sync=true")
        assert response.status_code == 200

    def test_sum(self):
        response = self.client.post("/scoring/sum?force_sync=true")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
