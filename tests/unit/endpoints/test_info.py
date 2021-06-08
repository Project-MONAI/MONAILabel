import unittest

from .context import BasicEndpointTestSuite


class EndPointInfo(BasicEndpointTestSuite):
    def test_info(self):
        response = self.client.get("/info/")
        assert response.status_code == 200

        # check if following fields exist in the response
        res = response.json()
        for f in ["version", "name", "labels"]:
            assert res[f]


if __name__ == "__main__":
    unittest.main()
