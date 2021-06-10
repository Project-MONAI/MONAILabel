import unittest

from .context import BasicEndpointTestSuite


class EndPointScoring(BasicEndpointTestSuite):
    def test_dice(self):
        response = self.client.post("/scoring/dice?run_sync=true")
        assert response.status_code == 200

    def test_sum(self):
        response = self.client.post("/scoring/sum?run_sync=true")
        assert response.status_code == 200

    def test_status(self):
        self.client.get("/scoring/")

    def test_stop(self):
        self.client.delete("/scoring/")


if __name__ == "__main__":
    unittest.main()
