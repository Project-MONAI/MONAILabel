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

import base64
import hashlib
import hmac
import json
import time
import unittest
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt import InvalidTokenError

from monailabel.config import settings
from monailabel.endpoints.user.auth import from_token


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _pem(key):
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    return private, public.decode()


class TestTokenValidation(unittest.TestCase):
    """With authentication enabled, only tokens signed by the realm's RSA key are accepted."""

    @classmethod
    def setUpClass(cls):
        cls.private, cls.public = _pem(rsa.generate_private_key(public_exponent=65537, key_size=2048))
        cls.other_private, _ = _pem(rsa.generate_private_key(public_exponent=65537, key_size=2048))

    def setUp(self):
        for ctx in (
            patch.object(settings, "MONAI_LABEL_AUTH_ENABLE", True),
            patch("monailabel.endpoints.user.auth.get_public_key", return_value=self.public),
        ):
            ctx.start()
            self.addCleanup(ctx.stop)

    @staticmethod
    def _claims(**extra):
        claims = {
            "preferred_username": "annotator1",
            "email": "a@example.org",
            "name": "An Notator",
            "realm_access": {"roles": [settings.MONAI_LABEL_AUTH_ROLE_ANNOTATOR]},
            "exp": int(time.time()) + 600,
        }
        claims.update(extra)
        return claims

    def test_valid_token_is_accepted(self):
        for alg in ("RS256", "RS384", "RS512"):
            token = jwt.encode(self._claims(), self.private, algorithm=alg)
            user = from_token(token)
            self.assertEqual(user.username, "annotator1")
            self.assertEqual(user.roles, [settings.MONAI_LABEL_AUTH_ROLE_ANNOTATOR])

    def test_token_signed_by_another_key_is_rejected(self):
        with self.assertRaises(InvalidTokenError):
            from_token(jwt.encode(self._claims(), self.other_private, algorithm="RS256"))

    def test_expired_token_is_rejected(self):
        with self.assertRaises(InvalidTokenError):
            from_token(jwt.encode(self._claims(exp=int(time.time()) - 10), self.private, algorithm="RS256"))

    def test_unsigned_token_is_rejected(self):
        header = _b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
        payload = _b64(json.dumps(self._claims()).encode())
        for token in (f"{header}.{payload}.", f"{header}.{payload}"):
            with self.assertRaises(InvalidTokenError):
                from_token(token)

    def test_token_signed_with_the_public_key_as_hmac_secret_is_rejected(self):
        header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        payload = _b64(json.dumps(self._claims()).encode())
        signature = hmac.new(self.public.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
        with self.assertRaises(InvalidTokenError):
            from_token(f"{header}.{payload}.{_b64(signature)}")

    def test_garbage_is_rejected(self):
        for token in ("", "abc", "a.b.c"):
            with self.assertRaises(InvalidTokenError):
                from_token(token)


if __name__ == "__main__":
    unittest.main()
