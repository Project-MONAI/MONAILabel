import unittest

from .context import BasicEndpointTestSuite


class EndPointActiveLearning(BasicEndpointTestSuite):
    def test_strategy(self):
        response = self.client.post("/activelearning/random")
        assert response.status_code == 200

        # check if following fields exist in the response
        res = response.json()
        for f in ["id", "name", "url"]:
            assert res[f]


if __name__ == "__main__":
    unittest.main()
