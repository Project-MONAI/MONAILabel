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

import cgi
import http.client
import json
import logging
import mimetypes
import os
import re
import ssl
import tempfile
from pathlib import Path
from urllib.parse import quote_plus, unquote, urlencode, urlparse

import requests

logger = logging.getLogger(__name__)


class MONAILabelClient:
    def __init__(self, server_url, tmpdir=None, client_id=None, username=None, password=None):
        self._server_url = server_url.rstrip("/").strip()
        self._tmpdir = tmpdir if tmpdir else tempfile.tempdir
        self._client_id = client_id
        self._headers = {}

    def _update_client_id(self, params):
        if params:
            params["client_id"] = self._client_id
        else:
            params = {"client_id": self._client_id}
        return params

    def update_auth(self, token):
        if token:
            self._headers["Authorization"] = f"{token['token_type']} {token['access_token']}"

    def get_server_url(self):
        return self._server_url

    def set_server_url(self, server_url):
        self._server_url = server_url.rstrip("/").strip()

    def auth_enabled(self) -> bool:
        selector = "/auth/"
        status, response, _, _ = MONAILabelUtils.http_method("GET", self._server_url, selector)
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {response}", status, response
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logger.debug(f"Response: {response}")
        enabled = json.loads(response).get("enabled", False)
        return True if enabled else False

    def auth_token(self, username, password):
        selector = "/auth/token"
        data = urlencode({"username": username, "password": password, "grant_type": "password"})
        status, response, _, _ = MONAILabelUtils.http_method(
            "POST", self._server_url, selector, data, None, "application/x-www-form-urlencoded"
        )
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {response}", status, response
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logger.debug(f"Response: {response}")
        return json.loads(response)

    def auth_valid_token(self) -> bool:
        selector = "/auth/token/valid"
        status, _, _, _ = MONAILabelUtils.http_method("GET", self._server_url, selector, headers=self._headers)
        return True if status == 200 else False

    def info(self):
        selector = "/info/"
        status, response, _, _ = MONAILabelUtils.http_method("GET", self._server_url, selector, headers=self._headers)
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {response}", status, response
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def next_sample(self, strategy, params):
        params = self._update_client_id(params)
        selector = f"/activelearning/{MONAILabelUtils.urllib_quote_plus(strategy)}"
        status, response, _, _ = MONAILabelUtils.http_method(
            "POST", self._server_url, selector, params, headers=self._headers
        )
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {response}", status, response
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def create_session(self, image_in, params=None):
        selector = "/session/"
        params = self._update_client_id(params)

        status, response, _ = MONAILabelUtils.http_upload(
            "PUT", self._server_url, selector, params, [image_in], headers=self._headers
        )
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {response}", status, response
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def get_session(self, session_id):
        selector = f"/session/{MONAILabelUtils.urllib_quote_plus(session_id)}"
        status, response, _, _ = MONAILabelUtils.http_method("GET", self._server_url, selector, headers=self._headers)
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {response}", status, response
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def remove_session(self, session_id):
        selector = f"/session/{MONAILabelUtils.urllib_quote_plus(session_id)}"
        status, response, _, _ = MONAILabelUtils.http_method(
            "DELETE", self._server_url, selector, headers=self._headers
        )
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {response}", status, response
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def upload_image(self, image_in, image_id=None, params=None):
        selector = f"/datastore/?image={MONAILabelUtils.urllib_quote_plus(image_id)}"

        files = {"file": image_in}
        params = self._update_client_id(params)
        fields = {"params": json.dumps(params) if params else "{}"}

        status, response, _, _ = MONAILabelUtils.http_multipart(
            "PUT", self._server_url, selector, fields, files, headers=self._headers
        )
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {response}",
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def save_label(self, image_in, label_in, tag="", params=None):
        selector = f"/datastore/label?image={MONAILabelUtils.urllib_quote_plus(image_in)}"
        if tag:
            selector += f"&tag={MONAILabelUtils.urllib_quote_plus(tag)}"

        params = self._update_client_id(params)
        fields = {
            "params": json.dumps(params),
        }
        files = {"label": label_in}

        status, response, _, _ = MONAILabelUtils.http_multipart(
            "PUT", self._server_url, selector, fields, files, headers=self._headers
        )
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {response}",
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def datastore(self):
        selector = "/datastore/?output=all"
        status, response, _, _ = MONAILabelUtils.http_method("GET", self._server_url, selector, headers=self._headers)
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {response}", status, response
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def download_label(self, label_id, tag):
        selector = "/datastore/label?label={}&tag={}".format(
            MONAILabelUtils.urllib_quote_plus(label_id), MONAILabelUtils.urllib_quote_plus(tag)
        )
        status, response, _, headers = MONAILabelUtils.http_method(
            "GET", self._server_url, selector, headers=self._headers
        )
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {response}", status, response
            )

        content_disposition = headers.get("content-disposition")

        if not content_disposition:
            logging.warning("Filename not found. Fall back to no loaded labels")
        file_name = MONAILabelUtils.get_filename(content_disposition)

        file_ext = "".join(Path(file_name).suffixes)
        local_filename = tempfile.NamedTemporaryFile(dir=self._tmpdir, suffix=file_ext).name
        with open(local_filename, "wb") as f:
            f.write(response)

        return local_filename

    def infer(self, model, image_in, params, label_in=None, file=None, session_id=None):
        selector = "/infer/{}?image={}".format(
            MONAILabelUtils.urllib_quote_plus(model),
            MONAILabelUtils.urllib_quote_plus(image_in),
        )
        if session_id:
            selector += f"&session_id={MONAILabelUtils.urllib_quote_plus(session_id)}"

        params = self._update_client_id(params)
        fields = {"params": json.dumps(params) if params else "{}"}
        files = {"label": label_in} if label_in else {}
        files.update({"file": file} if file and not session_id else {})

        status, form, files, _ = MONAILabelUtils.http_multipart(
            "POST", self._server_url, selector, fields, files, headers=self._headers
        )
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {form}",
            )

        form = json.loads(form) if isinstance(form, str) else form
        params = form.get("params") if files else form
        params = json.loads(params) if isinstance(params, str) else params

        image_out = MONAILabelUtils.save_result(files, self._tmpdir)
        return image_out, params

    def train_start(self, model, params):
        params = self._update_client_id(params)

        selector = "/train/"
        if model:
            selector += MONAILabelUtils.urllib_quote_plus(model)

        status, response, _, _ = MONAILabelUtils.http_method(
            "POST", self._server_url, selector, params, headers=self._headers
        )
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {response}",
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def train_stop(self):
        selector = "/train/"
        status, response, _, _ = MONAILabelUtils.http_method(
            "DELETE", self._server_url, selector, headers=self._headers
        )
        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {response}",
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def train_status(self, check_if_running=False):
        selector = "/train/"
        if check_if_running:
            selector += "?check_if_running=true"
        status, response, _, _ = MONAILabelUtils.http_method("GET", self._server_url, selector, headers=self._headers)
        if check_if_running:
            return status == 200

        if status != 200:
            raise MONAILabelException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {response}",
            )

        response = response.decode("utf-8") if isinstance(response, bytes) else response
        logging.debug(f"Response: {response}")
        return json.loads(response)


class MONAILabelError:
    RESULT_NOT_FOUND = 1
    SERVER_ERROR = 2
    SESSION_EXPIRED = 3
    UNKNOWN = 4


class MONAILabelException(Exception):
    def __init__(self, error, msg, status_code=None, response=None):
        self.error = error
        self.msg = msg
        self.status_code = status_code
        self.response = response


class MONAILabelUtils:
    @staticmethod
    def http_method(method, server_url, selector, body=None, headers=None, content_type=None):
        logging.debug(f"{method} {server_url}{selector}")

        parsed = urlparse(server_url)
        path = parsed.path.rstrip("/")
        selector = path + "/" + selector.lstrip("/")
        logging.debug(f"URI Path: {selector}")

        parsed = urlparse(server_url)
        if parsed.scheme == "https":
            logger.debug("Using HTTPS mode")
            # noinspection PyProtectedMember
            conn = http.client.HTTPSConnection(parsed.hostname, parsed.port, context=ssl._create_unverified_context())
        else:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port)

        headers = headers if headers else {}
        if body:
            if not content_type:
                if isinstance(body, dict):
                    body = json.dumps(body)
                    content_type = "application/json"
                else:
                    content_type = "text/plain"
            headers.update({"content-type": content_type, "content-length": str(len(body))})

        conn.request(method, selector, body=body, headers=headers)
        return MONAILabelUtils.send_response(conn)

    @staticmethod
    def http_upload(method, server_url, selector, fields, files, headers=None):
        logging.debug(f"{method} {server_url}{selector}")

        url = server_url.rstrip("/") + "/" + selector.lstrip("/")
        logging.debug(f"URL: {url}")

        files = [("files", (os.path.basename(f), open(f, "rb"))) for f in files]
        headers = headers if headers else {}
        response = (
            requests.post(url, files=files, headers=headers)
            if method == "POST"
            else requests.put(url, files=files, data=fields, headers=headers)
        )
        return response.status_code, response.text, None

    @staticmethod
    def http_multipart(method, server_url, selector, fields, files, headers={}):
        logging.debug(f"{method} {server_url}{selector}")

        content_type, body = MONAILabelUtils.encode_multipart_formdata(fields, files)
        headers = headers if headers else {}
        headers.update({"content-type": content_type, "content-length": str(len(body))})

        parsed = urlparse(server_url)
        path = parsed.path.rstrip("/")
        selector = path + "/" + selector.lstrip("/")
        logging.debug(f"URI Path: {selector}")

        if parsed.scheme == "https":
            logger.debug("Using HTTPS mode")
            # noinspection PyProtectedMember
            conn = http.client.HTTPSConnection(parsed.hostname, parsed.port, context=ssl._create_unverified_context())
        else:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port)

        conn.request(method, selector, body, headers)
        return MONAILabelUtils.send_response(conn, content_type)

    @staticmethod
    def send_response(conn, content_type="application/json"):
        response = conn.getresponse()
        logging.debug(f"HTTP Response Code: {response.status}")
        logging.debug(f"HTTP Response Message: {response.reason}")
        logging.debug(f"HTTP Response Headers: {response.getheaders()}")

        response_content_type = response.getheader("content-type", content_type)
        logging.debug(f"HTTP Response Content-Type: {response_content_type}")

        if "multipart" in response_content_type:
            if response.status == 200:
                form, files = MONAILabelUtils.parse_multipart(response.fp if response.fp else response, response.msg)
                logging.debug(f"Response FORM: {form}")
                logging.debug(f"Response FILES: {files.keys()}")
                return response.status, form, files, response.headers
            else:
                return response.status, response.read(), None, response.headers

        logging.debug("Reading status/content from simple response!")
        return response.status, response.read(), None, response.headers

    @staticmethod
    def save_result(files, tmpdir):
        for name in files:
            data = files[name]
            result_file = os.path.join(tmpdir, name)

            logging.debug(f"Saving {name} to {result_file}; Size: {len(data)}")
            dir_path = os.path.dirname(os.path.realpath(result_file))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            with open(result_file, "wb") as f:
                if isinstance(data, bytes):
                    f.write(data)
                else:
                    f.write(data.encode("utf-8"))

            # Currently only one file per response supported
            return result_file

    @staticmethod
    def encode_multipart_formdata(fields, files):
        limit = "----------lImIt_of_THE_fIle_eW_$"
        lines = []
        for key, value in fields.items():
            lines.append("--" + limit)
            lines.append('Content-Disposition: form-data; name="%s"' % key)
            lines.append("")
            lines.append(value)
        for key, filename in files.items():
            lines.append("--" + limit)
            lines.append(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"')
            lines.append("Content-Type: %s" % MONAILabelUtils.get_content_type(filename))
            lines.append("")
            with open(filename, mode="rb") as f:
                data = f.read()
                lines.append(data)
        lines.append("--" + limit + "--")
        lines.append("")

        body = bytearray()
        for line in lines:
            body.extend(line if isinstance(line, bytes) else line.encode("utf-8"))
            body.extend(b"\r\n")

        content_type = "multipart/form-data; boundary=%s" % limit
        return content_type, body

    @staticmethod
    def get_content_type(filename):
        return mimetypes.guess_type(filename)[0] or "application/octet-stream"

    @staticmethod
    def parse_multipart(fp, headers):
        fs = cgi.FieldStorage(
            fp=fp,
            environ={"REQUEST_METHOD": "POST"},
            headers=headers,
            keep_blank_values=True,
        )
        form = {}
        files = {}
        if hasattr(fs, "list") and isinstance(fs.list, list):
            for f in fs.list:
                logger.debug(f"FILE-NAME: {f.filename}; NAME: {f.name}; SIZE: {len(f.value)}")
                if f.filename:
                    files[f.filename] = f.value
                else:
                    form[f.name] = f.value
        return form, files

    @staticmethod
    def urllib_quote_plus(s):
        return quote_plus(s)

    @staticmethod
    def get_filename(content_disposition):
        file_name = re.findall(r"filename\*=([^;]+)", content_disposition, flags=re.IGNORECASE)
        if not file_name:
            file_name = re.findall('filename="(.+)"', content_disposition, flags=re.IGNORECASE)
        if "utf-8''" in file_name[0].lower():
            file_name = re.sub("utf-8''", "", file_name[0], flags=re.IGNORECASE)
            file_name = unquote(file_name)
        else:
            file_name = file_name[0]
        return file_name
