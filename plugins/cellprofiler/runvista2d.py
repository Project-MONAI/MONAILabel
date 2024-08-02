#################################
#
# Imports from useful Python libraries
#
#################################

import numpy
import os
import monai
import skimage
import importlib.metadata
import subprocess
import uuid
import shutil
import tempfile
import logging
import sys

import cgi
import http.client
import json
import mimetypes
import re
import ssl
from pathlib import Path
from urllib.parse import quote_plus, unquote, urlencode, urlparse

import requests

#################################
#
# Imports from CellProfiler
#
##################################

from cellprofiler_core.image import Image
from cellprofiler_core.module.image_segmentation import ImageSegmentation
from cellprofiler_core.object import Objects
from cellprofiler_core.setting import Binary, ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.preferences import get_default_output_directory
from cellprofiler_core.setting.text import Text, URL

CUDA_LINK = "https://pytorch.org/get-started/locally/"
Cellpose_link = " https://doi.org/10.1038/s41592-020-01018-x"
Omnipose_link = "https://doi.org/10.1101/2021.11.03.467199"
LOGGER = logging.getLogger(__name__)

__doc__ = f"""\
RunCellpose
===========

**RunCellpose** uses a pre-trained machine learning model (Cellpose) to detect cells or nuclei in an image.

This module is useful for automating simple segmentation tasks in CellProfiler.
The module accepts greyscale input images and produces an object set. Probabilities can also be captured as an image.

Loading in a model will take slightly longer the first time you run it each session. When evaluating
performance you may want to consider the time taken to predict subsequent images.

This module now also supports Ominpose. Omnipose builds on Cellpose, for the purpose of **RunCellpose** it adds 2 additional
features: additional models; bact-omni and cyto2-omni which were trained using the Omnipose architechture, and bact
and the mask reconstruction algorithm for Omnipose that was created to solve over-segemnation of large cells; useful for bacterial cells,
but can be used for other arbitrary and anisotropic shapes. You can mix and match Omnipose models with Cellpose style masking or vice versa.

The module has been updated to be compatible with the latest release of Cellpose. From the old version of the module the 'cells' model corresponds to 'cyto2' model.

Installation:

It is necessary that you have installed Cellpose version >= 1.0.2

You'll want to run `pip install cellpose` on your CellProfiler Python environment to setup Cellpose. If you have an older version of Cellpose
run 'python -m pip install cellpose --upgrade'.

To use Omnipose models, and mask reconstruction method you'll want to install Omnipose 'pip install omnipose' and Cellpose version 1.0.2 'pip install cellpose==1.0.2'.

On the first time loading into CellProfiler, Cellpose will need to download some model files from the internet. This
may take some time. If you want to use a GPU to run the model, you'll need a compatible version of PyTorch and a
supported GPU. Instructions are avaiable at this link: {CUDA_LINK}

Stringer, C., Wang, T., Michaelos, M. et al. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106 (2021). {Cellpose_link}
Kevin J. Cutler, Carsen Stringer, Paul A. Wiggins, Joseph D. Mougous. Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation. bioRxiv 2021.11.03.467199. {Omnipose_link}
============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""


def bytes_to_str(b):
    return b.decode("utf-8") if isinstance(b, bytes) else b


class MONAILabelClient:
    """
    Basic MONAILabel Client to invoke infer/train APIs over http/https
    """

    def __init__(self, server_url, tmpdir=None, client_id=None):
        """
        :param server_url: Server URL for MONAILabel (e.g. http://127.0.0.1:8000)
        :param tmpdir: Temp directory to save temporary files.  If None then it uses tempfile.tempdir
        :param client_id: Client ID that will be added for all basic requests
        """

        self._server_url = server_url.rstrip("/").strip()
        self._tmpdir = tmpdir if tmpdir else tempfile.tempdir if tempfile.tempdir else "/tmp"
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
        """
        Return server url

        :return: the url for monailabel server
        """
        return self._server_url

    def set_server_url(self, server_url):
        """
        Set url for monailabel server

        :param server_url: server url for monailabel
        """
        self._server_url = server_url.rstrip("/").strip()

    def auth_enabled(self) -> bool:
        """
        Check if Auth is enabled

        """
        selector = "/auth/"
        status, response, _, _ = MONAILabelUtils.http_method("GET", self._server_url, selector)
        if status != 200:
            return False

        response = bytes_to_str(response)
        LOGGER.debug(f"Response: {response}")
        enabled = json.loads(response).get("enabled", False)
        return True if enabled else False

    def auth_token(self, username, password):
        """
        Fetch Auth Token.  Currently only basic authentication is supported.

        :param username: UserName for basic authentication
        :param password: Password for basic authentication
        """
        selector = "/auth/token"
        data = urlencode({"username": username, "password": password, "grant_type": "password"})
        status, response, _, _ = MONAILabelUtils.http_method(
            "POST", self._server_url, selector, data, None, "application/x-www-form-urlencoded"
        )
        if status != 200:
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {bytes_to_str(response)}", status, response
            )

        response = bytes_to_str(response)
        LOGGER.debug(f"Response: {response}")
        return json.loads(response)

    def auth_valid_token(self) -> bool:
        selector = "/auth/token/valid"
        status, _, _, _ = MONAILabelUtils.http_method("GET", self._server_url, selector, headers=self._headers)
        return True if status == 200 else False

    def info(self):
        """
        Invoke /info/ request over MONAILabel Server

        :return: json response
        """
        selector = "/info/"
        status, response, _, _ = MONAILabelUtils.http_method("GET", self._server_url, selector, headers=self._headers)
        if status != 200:
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {bytes_to_str(response)}", status, response
            )

        response = bytes_to_str(response)
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def next_sample(self, strategy, params):
        """
        Get Next sample

        :param strategy: Name of strategy to be used for fetching next sample
        :param params: Additional JSON params as part of strategy request
        :return: json response which contains information about next image selected for annotation
        """
        params = self._update_client_id(params)
        selector = f"/activelearning/{MONAILabelUtils.urllib_quote_plus(strategy)}"
        status, response, _, _ = MONAILabelUtils.http_method(
            "POST", self._server_url, selector, params, headers=self._headers
        )
        if status != 200:
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {bytes_to_str(response)}", status, response
            )

        response = bytes_to_str(response)
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def create_session(self, image_in, params=None):
        """
        Create New Session

        :param image_in: filepath for image to be sent to server as part of session creation
        :param params: additional JSON params as part of session reqeust
        :return: json response which contains session id and other details
        """
        selector = "/session/"
        params = self._update_client_id(params)

        status, response, _ = MONAILabelUtils.http_upload(
            "PUT", self._server_url, selector, params, [image_in], headers=self._headers
        )
        if status != 200:
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {bytes_to_str(response)}", status, response
            )

        response = bytes_to_str(response)
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def get_session(self, session_id):
        """
        Get Session

        :param session_id: Session Id
        :return: json response which contains more details about the session
        """
        selector = f"/session/{MONAILabelUtils.urllib_quote_plus(session_id)}"
        status, response, _, _ = MONAILabelUtils.http_method("GET", self._server_url, selector, headers=self._headers)
        if status != 200:
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {bytes_to_str(response)}", status, response
            )

        response = bytes_to_str(response)
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def remove_session(self, session_id):
        """
        Remove any existing Session

        :param session_id: Session Id
        :return: json response
        """
        selector = f"/session/{MONAILabelUtils.urllib_quote_plus(session_id)}"
        status, response, _, _ = MONAILabelUtils.http_method(
            "DELETE", self._server_url, selector, headers=self._headers
        )
        if status != 200:
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {bytes_to_str(response)}", status, response
            )

        response = bytes_to_str(response)
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def upload_image(self, image_in, image_id=None, params=None):
        """
        Upload New Image to MONAILabel Datastore

        :param image_in: Image File Path
        :param image_id: Force Image ID;  If not provided then Server it auto generate new Image ID
        :param params: Additional JSON params
        :return: json response which contains image id and other details
        """
        selector = f"/datastore/?image={MONAILabelUtils.urllib_quote_plus(image_id)}"

        files = {"file": image_in}
        params = self._update_client_id(params)
        fields = {"params": json.dumps(params) if params else "{}"}

        status, response, _, _ = MONAILabelUtils.http_multipart(
            "PUT", self._server_url, selector, fields, files, headers=self._headers
        )
        if status != 200:
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {bytes_to_str(response)}",
            )

        response = bytes_to_str(response)
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def save_label(self, image_id, label_in, tag="", params=None):
        """
        Save/Submit Label

        :param image_id: Image Id for which label needs to saved/submitted
        :param label_in: Label File path which shall be saved/submitted
        :param tag: Save label against tag in datastore
        :param params: Additional JSON params for the request
        :return: json response
        """
        selector = f"/datastore/label?image={MONAILabelUtils.urllib_quote_plus(image_id)}"
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
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {bytes_to_str(response)}",
            )

        response = bytes_to_str(response)
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def datastore(self):
        selector = "/datastore/?output=all"
        status, response, _, _ = MONAILabelUtils.http_method("GET", self._server_url, selector, headers=self._headers)
        if status != 200:
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {bytes_to_str(response)}", status, response
            )

        response = bytes_to_str(response)
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
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR, f"Status: {status}; Response: {bytes_to_str(response)}", status, response
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

    def infer(self, model, image_id, params, label_in=None, file=None, session_id=None):
        """
        Run Infer

        :param model: Name of Model
        :param image_id: Image Id
        :param params: Additional configs/json params as part of Infer request
        :param label_in: File path for label mask which is needed to run Inference (e.g. In case of Scribbles)
        :param file: File path for Image (use raw image instead of image_id)
        :param session_id: Session ID (use existing session id instead of image_id)
        :return: response_file (label mask), response_body (json result/output params)
        """
        selector = "/infer/{}?image={}".format(
            MONAILabelUtils.urllib_quote_plus(model),
            MONAILabelUtils.urllib_quote_plus(image_id),
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
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {bytes_to_str(form)}",
            )

        form = json.loads(form) if isinstance(form, str) else form
        params = form.get("params") if files else form
        params = json.loads(params) if isinstance(params, str) else params

        image_out = MONAILabelUtils.save_result(files, self._tmpdir)
        return image_out, params

    def wsi_infer(self, model, image_id, body=None, output="dsa", session_id=None):
        """
        Run WSI Infer in case of Pathology App

        :param model: Name of Model
        :param image_id: Image Id
        :param body: Additional configs/json params as part of Infer request
        :param output: Output File format (dsa|asap|json)
        :param session_id: Session ID (use existing session id instead of image_id)
        :return: response_file (None), response_body
        """
        selector = "/infer/wsi/{}?image={}".format(
            MONAILabelUtils.urllib_quote_plus(model),
            MONAILabelUtils.urllib_quote_plus(image_id),
        )
        if session_id:
            selector += f"&session_id={MONAILabelUtils.urllib_quote_plus(session_id)}"
        if output:
            selector += f"&output={MONAILabelUtils.urllib_quote_plus(output)}"

        body = self._update_client_id(body if body else {})
        status, form, _, _ = MONAILabelUtils.http_method("POST", self._server_url, selector, body)
        if status != 200:
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {bytes_to_str(form)}",
            )

        return None, form

    def train_start(self, model, params):
        """
        Run Train Task

        :param model: Name of Model
        :param params: Additional configs/json params as part of Train request
        :return: json response
        """
        params = self._update_client_id(params)

        selector = "/train/"
        if model:
            selector += MONAILabelUtils.urllib_quote_plus(model)

        status, response, _, _ = MONAILabelUtils.http_method(
            "POST", self._server_url, selector, params, headers=self._headers
        )
        if status != 200:
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {bytes_to_str(response)}",
            )

        response = bytes_to_str(response)
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def train_stop(self):
        """
        Stop any running Train Task(s)

        :return: json response
        """
        selector = "/train/"
        status, response, _, _ = MONAILabelUtils.http_method(
            "DELETE", self._server_url, selector, headers=self._headers
        )
        if status != 200:
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {bytes_to_str(response)}",
            )

        response = bytes_to_str(response)
        logging.debug(f"Response: {response}")
        return json.loads(response)

    def train_status(self, check_if_running=False):
        """
        Check Train Task Status

        :param check_if_running: Fast mode.  Only check if training is Running
        :return: boolean if check_if_running is enabled; else json response that contains of full details
        """
        selector = "/train/"
        if check_if_running:
            selector += "?check_if_running=true"
        status, response, _, _ = MONAILabelUtils.http_method("GET", self._server_url, selector, headers=self._headers)
        if check_if_running:
            return status == 200

        if status != 200:
            raise MONAILabelClientException(
                MONAILabelError.SERVER_ERROR,
                f"Status: {status}; Response: {bytes_to_str(response)}",
            )

        response = bytes_to_str(response)
        logging.debug(f"Response: {response}")
        return json.loads(response)


class MONAILabelError:
    """
    Type of Inference Model

    Attributes:
        SERVER_ERROR -           Server Error
        SESSION_EXPIRED -        Session Expired
        UNKNOWN -                Unknown Error
    """

    SERVER_ERROR = 1
    SESSION_EXPIRED = 2
    UNKNOWN = 3


class MONAILabelClientException(Exception):
    """
    MONAILabel Client Exception
    """

    __slots__ = ["error", "msg"]

    def __init__(self, error, msg, status_code=None, response=None):
        """
        :param error: Error code represented by MONAILabelError
        :param msg: Error message
        :param status_code: HTTP Response code
        :param response: HTTP Response
        """
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
            LOGGER.debug("Using HTTPS mode")
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
            LOGGER.debug("Using HTTPS mode")
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
                LOGGER.debug(f"FILE-NAME: {f.filename}; NAME: {f.name}; SIZE: {len(f.value)}")
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


class RunVISTA2D(ImageSegmentation):
    category = "Object Processing"

    module_name = "RunVISTA2D"

    variable_revision_number = 1

    doi = {
        "Please cite the following when using RunVISTA2D:": "https://doi.org/10.48550/arXiv.2406.05285",
    }

    def create_settings(self):
        super(RunVISTA2D, self).create_settings()

        self.server_address = URL(
            text="MONAI label server address",
            doc="""\
Please set up the MONAI label server in local/cloud environment and fill the server address here.
""",
        )

        self.model_name = Choice(
            text="The model for running the inference",
            choices=["cell_vista_segmentation"],
            value="cell_vista_segmentation",
            doc="""
Pick the model for running infernce. Now only VISTA2D is available.
"""
            )

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.server_address,
            self.model_name,
        ]
    
    def visible_settings(self):
        return [
            self.x_name,
            self.y_name,
            self.server_address,
            self.model_name,
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value
        images = workspace.image_set
        x = images.get_image(x_name)
        dimensions = x.dimensions
        x_data = x.pixel_data

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_img_dir = os.path.join(temp_dir, "img")
            os.makedirs(temp_img_dir, exist_ok=True)
            temp_img_path = os.path.join(temp_img_dir, x_name+".tiff")
            temp_mask_dir = os.path.join(temp_dir, "mask")
            os.makedirs(temp_mask_dir, exist_ok=True)
            skimage.io.imsave(temp_img_path, x_data)
            monailabel_client = MONAILabelClient(server_url=self.server_address.value, tmpdir=temp_mask_dir)
            image_out, params = monailabel_client.infer(model=self.model_name.value, image_id="", params={}, file=temp_img_path)
            print(f"Image out:\n{image_out}")
            print(f"Params:\n{params}")
            y_data = skimage.io.imread(image_out)
            


        y = Objects()
        y.segmented = y_data
        y.parent_image = x.parent_image
        objects = workspace.object_set
        objects.add_objects(y, y_name)


        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data
            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure):
        layout = (2, 1)
        figure.set_subplots(
            dimensions=workspace.display_data.dimensions, subplots=layout
        )

        figure.subplot_imshow(
            colormap="gray",
            image=workspace.display_data.x_data,
            title="Input Image",
            x=0,
            y=0,
        )

        figure.subplot_imshow_labels(
            image=workspace.display_data.y_data,
            sharexy=figure.subplot(0, 0),
            title=self.y_name.value,
            x=1,
            y=0,
        )

    # def upgrade_settings(self, setting_values, variable_revision_number, module_name):
    #     ...
