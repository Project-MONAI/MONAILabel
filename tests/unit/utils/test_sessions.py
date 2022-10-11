# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import unittest

from monailabel.utils.sessions import Sessions


class MyTestCase(unittest.TestCase):
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    data_dir = os.path.join(base_dir, "tests", "data")
    download_dir = os.path.join(base_dir, "tests", "downloads")

    def test_auto_expiry(self):
        sessions = Sessions(os.path.join(self.data_dir, "sessions"))
        count = len(sessions)

        session = sessions.add_session(os.path.join(self.download_dir, "dataset.zip"), uncompress=True, expiry=1)
        assert session

        time.sleep(2)
        sessions.remove_expired()
        assert len(sessions) == count


if __name__ == "__main__":
    unittest.main()
