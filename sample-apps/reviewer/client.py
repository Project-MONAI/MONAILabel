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

"""
Lightweight MONAILabel client for Reviewer workflow.
Simplified client focused only on review operations.

Reuses monailabel.client.MONAILabelClient but routes only review endpoints.
Does NOT load AI models, no inference, no active learning.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class LightweightReviewClient:
    """
    Simplified client for reviewing segmentations.
    Only offers review-related endpoints, no AI operation endpoints.

    This client provides:
    - List images
    - Download image data
    - Download label/segmentation (any version)
    - Download label metadata
    - Update label metadata
    - Save label

    Does NOT provide:
    - AI model inference
    - Training endpoints
    - Active learning
    - Segmentation tasks
    """

    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize lightweight review client.

        Parameters:
        - server_url: MONAI Label server URL
        - timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }

        # Verify server connectivity
        self.ping()

        logger.info(f"ReviewClient initialized: {self.server_url}")

    def ping(self) -> bool:
        """Check if server is reachable."""
        try:
            response = requests.get(
                f"{self.server_url}",
                timeout=5
            )
            status = response.status_code == 200
            if status:
                logger.info(f"✓ Connected to server: {self.server_url}")
            else:
                logger.warning(f"✗ Server responded with status: {response.status_code}")
            return status
        except Exception as e:
            logger.warning(f"✗ Cannot connect to server: {e}")
            return False

    def list_images(
        self,
        offset: int = 0,
        limit: int = 100,
        status_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all images available for review.

        Query Parameters:
        - offset: Pagination offset (default: 0)
        - limit: Pagination limit (default: 100)
        - status_filter: Filter by status (approved/flagged/unapproved)

        Returns:
        {
            "summary": {
                "total": 100,
                "approved": 45,
                "flagged": 5,
                "pending": 50
            },
            "results": [...list of image metadata...],
            "metadata": {...}
        }
        """
        try:
            params = {
                "limit": str(limit),
                "offset": str(offset)
            }
            if status_filter:
                params["status_filter"] = status_filter

            response = requests.get(
                f"{self.server_url}/review/cases",
                params=params,
                timeout=self.timeout,
                headers=self.headers
            )

            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Listed {len(data.get('results', []))} images")
                return data
            else:
                logger.error(f"Failed to list images: {response.status_code} - {response.text}")
                return {"status": "error", "error": f"Server error {response.status_code}"}

        except Exception as e:
            logger.error(f"Error listing images: {e}")
            return {"status": "error", "error": str(e)}

    def download_image(self, image_id: str) -> bytes:
        """
        Download DICOM image data.

        Parameters:
        - image_id: Unique identifier for the image

        Returns:
        Image file bytes in original format
        """
        try:
            response = requests.get(
                f"{self.server_url}/datastore/image",
                params={"image": image_id},
                timeout=self.timeout
            )

            if response.status_code == 200:
                image_data = response.content
                logger.debug(f"Downloaded image: {image_id} ({len(image_data)} bytes)")
                return image_data
            else:
                logger.error(f"Failed to download image {image_id}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error downloading image {image_id}: {e}")
            return None

    def download_label(
        self,
        label_id: str,
        tag: str = "final"
    ) -> Dict[str, Any]:
        """
        Download segmentation label or mask.

        Parameters:
        - label_id: Label/segmentation ID
        - tag: Label version/tag (default: final)

        Returns:
        {
            "status": "success",
            "label_id": "label_001",
            "tag": "final",
            "data": {binary representation},
            "metadata": {...}
        }
        """
        try:
            response = requests.get(
                f"{self.server_url}/datastore/label",
                params={"label": label_id, "tag": tag},
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Downloaded label: {label_id} (tag={tag})")
                return data
            else:
                logger.warning(f"Label {label_id} (tag={tag}) not found: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error downloading label {label_id}: {e}")
            return None

    def download_labelinfo(self, label_id: str) -> Dict[str, Any]:
        """
        Download label metadata.

        Parameters:
        - label_id: Label/segmentation ID

        Returns:
        {
            "status": "success",
            "label_id": "label_001",
            "info": {...metadata...}
        }
        """
        try:
            response = requests.get(
                f"{self.server_url}/datastore/label/info",
                params={"label": label_id, "tag": "final"},
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Downloaded label info: {label_id}")
                return data
            else:
                return None

        except Exception as e:
            logger.error(f"Error downloading label info for {label_id}: {e}")
            return None

    def update_labelinfo(
        self,
        label_id: str,
        status: str,
        level: Optional[str] = None,
        comment: Optional[str] = None,
        reviewer_name: Optional[str] = None,
        reviewer_email: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> bool:
        """
        Update label metadata for review.

        Parameters:
        - label_id: Label/segmentation ID
        - status: "approved" | "flagged" | "unapproved"
        - level: "easy" | "medium" | "hard"
        - comment: Review comment text
        - reviewer_name: Name of reviewer
        - reviewer_email: Email of reviewer
        - workflow_id: Optional workflow identifier

        Returns:
        True on success, False on failure
        """
        try:
            review_info = {
                "status": status,
                "reviewer_name": reviewer_name,
                "workflow_id": workflow_id
            }

            if level:
                review_info["level"] = level
            if comment:
                review_info["comment"] = comment
            if reviewer_email:
                review_info["reviewer_email"] = reviewer_email
            if reviewer_name:
                review_info["reviewer"] = reviewer_name

            payload = {"info": json.dumps(review_info)}

            response = requests.put(
                f"{self.server_url}/datastore/label/info",
                params={"label": label_id, "tag": "final"},
                data=payload,
                headers=self.headers
            )

            if response.status_code == 200:
                logger.info(f"Updated label {label_id}: {status} - {reviewer_name}")
                return True
            else:
                logger.warning(f"Failed to update label {label_id}: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error updating label {label_id}: {e}")
            return False

    def save_label(
        self,
        image_id: str,
        label_file: Path,
        tag: str = "final",
        reviewer_name: str = "Reviewer",
        comment: Optional[str] = None,
        version_note: Optional[str] = None
    ) -> bool:
        """
        Save a new or updated segmentation label.

        Parameters:
        - image_id: Image ID
        - label_file: Path to segmentation file
        - tag: Label version/tag
        - reviewer_name: Name of reviewer/saver
        - comment: Optional comment about this version
        - version_note: Reason for this version

        Returns:
        True on success, False on failure
        """
        try:
            # Prepare approvals data
            approvals = {
                "reviewer_name": reviewer_name
            }
            if comment:
                approvals["comment"] = comment
            if version_note:
                approvals["comment"] = f"{version_note}: {comment or ''}"

            params_payload = json.dumps(approvals)

            with open(label_file, "rb") as f:
                files = {"label": (image_id, f)}

                response = requests.put(
                    f"{self.server_url}/datastore/label",
                    params={
                        "image": image_id,
                        "tag": tag
                    },
                    data={"params": params_payload},
                    files=files,
                    headers={"Accept": "application/json"}
                )

                if response.status_code == 200:
                    logger.info(f"Saved label for {image_id} (tag={tag})")
                    return True
                else:
                    logger.warning(f"Failed to save label for {image_id}: {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"Error saving label for {image_id}: {e}")
            return False

    def get_versions(self, image_id: str) -> Dict[str, Any]:
        """
        List all available versions for an image.

        Parameters:
        - image_id: Image ID

        Returns:
        {
            "status": "success",
            "image_id": "CT_abdomen_001",
            "versions": [...]
        }
        """
        try:
            response = requests.get(
                f"{self.server_url}/review/versions",
                params={"image": image_id},
                timeout=self.timeout,
                headers=self.headers
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.error(f"Error getting versions for {image_id}: {e}")
            return {"status": "error", "error": str(e)}

    def generate_report(self, fmt: str = "json", reviewer: str = None) -> Dict[str, Any]:
        """
        Generate a review summary report.

        Parameters:
        - fmt: Report format ("json", "csv", "html")
        - reviewer: Optional reviewer filter

        Returns:
        Report data
        """
        try:
            response = requests.get(
                f"{self.server_url}/review/report",
                params={"fmt": fmt, "reviewer": reviewer},
                timeout=self.timeout,
                headers=self.headers
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"status": "error", "error": str(e)}

    def info(self) -> Dict[str, Any]:
        """
        Get server information.

        Returns:
        Server info including configuration
        """
        try:
            response = requests.get(
                f"{self.server_url}",
                timeout=5,
                headers=self.headers
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"error": str(e)}


# Convenience functions
def create_client(server_url: Optional[str] = None) -> LightweightReviewClient:
    """
    Create and return a lightweight review client.

    Parameters:
    - server_url: Optional server URL override

    Returns:
    LightweightReviewClient instance
    """
    return LightweightReviewClient(server_url=server_url)
