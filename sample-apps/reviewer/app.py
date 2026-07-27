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
Lightweight MONAILabel app for review workflow.

This app exposes no AI tasks and relies on the standard MONAILabel datastore
plus reviewer-specific aggregate endpoints.
"""
import logging
import os
from typing import Any, Dict

from monailabel.interfaces.app import MONAILabelApp

logger = logging.getLogger(__name__)


class ReviewerApp(MONAILabelApp):
    """
    Lightweight review-specific MONAILabel application.

    This app is designed for:
    - Reviewing and validating AI-generated segmentations
    - Managing review metadata (comments, ratings, approvals)
    - Generating review reports
    - Simple cache management
    - NO AI model training or inference

    The app provides a simplified focus on review operations
    without the overhead of AI model management.
    """

    def __init__(self, app_dir: str, studies: str, conf: Dict):
        """
        Initialize the reviewer app.

        Parameters:
        - app_dir: Root path of the app
        - studies: Datastore path
        - conf: Configuration dictionary
        """
        self.review_config = self._load_review_config(studies, conf)

        super().__init__(
            app_dir=app_dir,
            studies=self.review_config["studies"],
            conf=conf,
            name="MONAILabel Reviewer",
            description="Lightweight review app - validates AI-generated segmentations",
            version="1.0.0",
        )

        logger.info(f"MONAILabel Reviewer App initialized")
        logger.info(f"Mode: {self.review_config['mode']}")
        logger.info(f"Server: {self.review_config['server_url']}")
        logger.info(f"Studies: {self.review_config['studies']}")

    def _load_review_config(self, studies: str, conf: Dict) -> Dict[str, Any]:
        """Load review-specific configuration."""
        conf_dict = conf or {}

        config = {
            "server_url": conf_dict.get("server_url") or conf_dict.get("monailabel_server"),
            "studies": studies or conf_dict.get("studies") or conf_dict.get("datastore"),
            "reviewer_name": conf_dict.get("reviewer_name"),
            "reviewer_email": conf_dict.get("reviewer_email"),
            "mode": conf_dict.get("mode", "review"),
        }

        # Override with environment variables
        config["server_url"] = (
            config["server_url"] or
            os.environ.get("MONAI_LABEL_SERVER", "http://localhost:8000")
        )
        config["studies"] = (
            config["studies"] or
            os.environ.get("MONAI_LABEL_STUDIES", "")
        )
        config["reviewer_name"] = (
            config["reviewer_name"] or
            os.environ.get("MONAI_LABEL_REVIEWER_NAME", "Reviewer")
        )
        config["reviewer_email"] = (
            config["reviewer_email"] or
            os.environ.get("MONAI_LABEL_REVIEWER_EMAIL", "")
        )
        config["mode"] = os.environ.get("MONAI_LABEL_REVIEW_MODE", config["mode"])

        return config

    def init_infers(self) -> Dict[str, Any]:
        return {}

    def init_trainers(self) -> Dict[str, Any]:
        return {}

    def init_strategies(self) -> Dict[str, Any]:
        return {}

    def init_scoring_methods(self) -> Dict[str, Any]:
        return {}

    def info(self) -> Dict[str, Any]:
        """
        Get application information.

        Returns:
        {
            "name": "MONAILabel Reviewer",
            "description": "Lightweight review app",
            "version": "1.0.0",
            "studies": "/path/to/images",
            "config": {...}
        }
        """
        meta = super().info()
        meta.update(
            {
                "studies": self.review_config.get("studies"),
                "config": self.review_config,
                "features": [
                    "Image listing and browsing",
                    "Segmentation download",
                    "Metadata management",
                    "Review status tracking",
                    "Version control",
                    "Report generation",
                ],
                "mode": "REVIEW ONLY - No AI models loaded",
            }
        )
        return meta

    def allowed_keys(self) -> list:
        """Get allowed configuration keys."""
        return ["server_url", "studies", "reviewer_name", "reviewer_email", "mode"]
