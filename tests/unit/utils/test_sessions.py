import os
import time
import unittest

from monailabel.utils.sessions import Sessions


class MyTestCase(unittest.TestCase):
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    data_dir = os.path.join(base_dir, "tests", "data")

    def test_auto_expiry(self):
        sessions = Sessions(os.path.join(self.data_dir, "sessions"))
        count = len(sessions)

        session = sessions.add_session(os.path.join(self.data_dir, "dataset.zip"), uncompress=True, expiry=1)
        assert session

        time.sleep(2)
        sessions.remove_expired()
        assert len(sessions) == count


if __name__ == "__main__":
    unittest.main()
