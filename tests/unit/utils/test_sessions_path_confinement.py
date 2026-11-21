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

import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
import zipfile

from monailabel.utils.sessions import Sessions, safe_unpack_archive


class TestSessionsPathConfinement(unittest.TestCase):
    """Archive members and session ids must resolve inside the extraction directory / session store."""

    def setUp(self):
        # everything, including the paths the archives try to reach, stays under self.base
        self.base = os.path.realpath(tempfile.mkdtemp(prefix="monailabel_sessions_confine_"))
        self.extract_dir = os.path.join(self.base, "a", "b", "extract")
        self.outside = os.path.join(self.base, "outside")
        self.store = os.path.join(self.base, "sessions")
        for d in (self.extract_dir, self.outside, self.store):
            os.makedirs(d)

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _tar(self, members, name="archive.tar", mode="w"):
        path = os.path.join(self.base, name)
        with tarfile.open(path, mode) as tf:
            for member, payload in members:
                info = tarfile.TarInfo(member)
                if isinstance(payload, bytes):
                    info.size = len(payload)
                    tf.addfile(info, io.BytesIO(payload))
                else:
                    info.type, info.linkname = payload
                    tf.addfile(info)
        return path

    def _files_outside_extract_dir(self):
        found = []
        for root, _, files in os.walk(self.base):
            if root.startswith(self.extract_dir) or root == self.base:
                continue
            found.extend(os.path.join(root, f) for f in files)
        return found

    def test_tar_rejects_member_with_parent_dirs(self):
        archive = self._tar([("ok.txt", b"ok"), ("../../../outside/pwned.txt", b"pwned")])
        with self.assertRaises(ValueError):
            safe_unpack_archive(archive, self.extract_dir)
        self.assertEqual(self._files_outside_extract_dir(), [])
        # nothing is written when any member is rejected
        self.assertEqual(os.listdir(self.extract_dir), [])

    def test_tar_gz_rejects_member_with_parent_dirs(self):
        archive = self._tar([("../../../outside/pwned.txt", b"pwned")], name="archive.tar.gz", mode="w:gz")
        with self.assertRaises(ValueError):
            safe_unpack_archive(archive, self.extract_dir)
        self.assertEqual(self._files_outside_extract_dir(), [])

    def test_tar_rejects_member_with_absolute_path(self):
        archive = self._tar([(os.path.join(self.outside, "pwned.txt"), b"pwned")])
        with self.assertRaises(ValueError):
            safe_unpack_archive(archive, self.extract_dir)
        self.assertEqual(self._files_outside_extract_dir(), [])

    def test_tar_rejects_symlink_member(self):
        archive = self._tar([("lnk", (tarfile.SYMTYPE, self.outside)), ("lnk/pwned.txt", b"pwned")])
        with self.assertRaises(ValueError):
            safe_unpack_archive(archive, self.extract_dir)
        self.assertEqual(self._files_outside_extract_dir(), [])
        self.assertFalse(os.path.lexists(os.path.join(self.extract_dir, "lnk")))

    def test_tar_rejects_hardlink_member(self):
        archive = self._tar([("lnk", (tarfile.LNKTYPE, os.path.join(self.outside, "x")))])
        with self.assertRaises(ValueError):
            safe_unpack_archive(archive, self.extract_dir)

    def test_tar_accepts_plain_members(self):
        archive = self._tar([("series/1.dcm", b"one"), ("series/2.dcm", b"two")])
        safe_unpack_archive(archive, self.extract_dir)
        with open(os.path.join(self.extract_dir, "series", "2.dcm"), "rb") as fp:
            self.assertEqual(fp.read(), b"two")

    def test_zip_keeps_members_inside(self):
        archive = os.path.join(self.base, "archive.zip")
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("series/1.dcm", "one")
            zf.writestr("../../../outside/pwned.txt", "pwned")
        safe_unpack_archive(archive, self.extract_dir)
        self.assertTrue(os.path.exists(os.path.join(self.extract_dir, "series", "1.dcm")))
        self.assertEqual(self._files_outside_extract_dir(), [])

    def test_rejects_unsupported_file(self):
        not_archive = os.path.join(self.base, "image.nii.gz")
        with open(not_archive, "wb") as fp:
            fp.write(b"not an archive")
        with self.assertRaises(ValueError):
            safe_unpack_archive(not_archive, self.extract_dir)

    def test_add_session_rejects_tar_with_parent_dirs(self):
        sessions = Sessions(self.store)
        archive = self._tar([("../../../outside/pwned.txt", b"pwned")])
        with self.assertRaises(ValueError):
            sessions.add_session(archive, uncompress=True)
        self.assertEqual(os.listdir(self.outside), [])
        self.assertEqual(os.listdir(self.store), [])

    def test_add_session_accepts_plain_tar(self):
        sessions = Sessions(self.store)
        archive = self._tar([("series/1.dcm", b"one")])
        session_id, info = sessions.add_session(archive, uncompress=True)
        self.assertTrue(os.path.exists(os.path.join(info.image, "series", "1.dcm")))
        self.assertEqual(sessions.get_session(session_id).name, session_id)

    def test_session_id_outside_store_is_not_resolved(self):
        sessions = Sessions(self.store)
        # a directory outside the store that looks like a dangling session
        victim = os.path.join(self.outside, "victim")
        os.makedirs(victim)
        with open(os.path.join(victim, "meta.info"), "w") as fp:
            fp.write(json.dumps({"name": "victim", "path": victim, "image": os.path.join(victim, "missing")}))

        for session_id in (os.path.relpath(victim, self.store), victim, "..", ".", "", "a/b", "a\\b"):
            self.assertIsNone(sessions.get_session(session_id))
            sessions.remove_session(session_id)
        self.assertTrue(os.path.exists(os.path.join(victim, "meta.info")))
        self.assertTrue(os.path.isdir(self.store))

        with self.assertRaises(ValueError):
            sessions.add_session(os.path.join(victim, "meta.info"), session_id="../outside/victim2")
        self.assertFalse(os.path.exists(os.path.join(self.outside, "victim2")))


if __name__ == "__main__":
    unittest.main()
