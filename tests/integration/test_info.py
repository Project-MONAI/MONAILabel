import unittest

import requests

from . import SERVER_URI


class EndPointInfo(unittest.TestCase):
    def test_info(self):
        response = requests.get(f"{SERVER_URI}/info/")
        assert response.status_code == 200

        # check if following fields exist in the response
        res = response.json()
        for f in ["version", "name", "labels"]:
            assert res[f]


if __name__ == "__main__":
    unittest.main()
