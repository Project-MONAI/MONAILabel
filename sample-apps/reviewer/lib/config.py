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
Review app configuration for lightweight MONAILabel reviewer.

This configuration is minimal - designed for review-only workflow
without heavy AI model dependencies.
"""
import os
from typing import Optional


class ReviewConfig:
    """
    Configuration for MONAILabel Reviewer application.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
        reviewer_name: Optional[str] = None,
        reviewer_email: Optional[str] = None,
        mode: str = "review",
    ):
        """
        Initialize reviewer configuration from arguments and environment variables.
        
        Parameters:
            server_url (Optional[str]): URL of the MONAI Label server.
            cache_dir (Optional[str]): Directory used for review data caching.
            reviewer_name (Optional[str]): Display name of the reviewer.
            reviewer_email (Optional[str]): Email address of the reviewer.
            mode (str): Operating mode, such as ``"review"`` or ``"standalone"``.
        """
        # Resolve server URL
        self.server_url = server_url
        if not self.server_url:
            # Use default if not provided
            self.server_url = os.environ.get("MONAI_LABEL_SERVER", "http://localhost:8000")

        # Resolve cache directory
        self.cache_dir = cache_dir
        if not self.cache_dir:
            default_cache_dir = os.path.join(os.path.expanduser("~"), "monailabel_reviewer", "cache")
            self.cache_dir = os.environ.get("MONAI_LABEL_REVIEW_CACHE", default_cache_dir)

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Reviewer info
        self.reviewer_name = reviewer_name
        if not self.reviewer_name:
            self.reviewer_name = os.environ.get("MONAI_LABEL_REVIEWER_NAME", "Reviewer")

        self.reviewer_email = reviewer_email
        if not self.reviewer_email:
            self.reviewer_email = os.environ.get("MONAI_LABEL_REVIEWER_EMAIL", "")

        # Operating mode
        self.mode = mode
        if mode == "standalone":
            self.review_only = True
            self.auto_sync = False
        else:
            self.review_only = False
            # In full review mode, can optionally sync to server
            self.auto_sync = os.environ.get("MONAI_LABEL_REVIEW_AUTO_SYNC", "false") == "true"

        # Metadata
        self.project_name = os.environ.get("MONAI_LABEL_PROJECT_NAME", "MONAILabel_Reviewer")
        self.workspace_dir = os.environ.get("MONAI_LABEL_STUDIES", "")

        # Review settings
        self.max_history = int(os.environ.get("MONAI_LABEL_REVIEW_MAX_HISTORY", "10"))
        self.cache_timeout = int(os.environ.get("MONAI_LABEL_REVIEW_CACHE_TIMEOUT", "3600"))

    @property
    def anaconda_channel(self) -> str:
        """Return the Anaconda channel name for deployment."""
        return os.environ.get("MONAI_LABEL_CONDA_CHANNEL", "projectmonai")

    @property
    def server_mode(self) -> str:
        """Describe whether the reviewer operates in standalone or server-synchronized mode.
        
        Returns:
            str: The current review mode description.
        """
        if self.review_only:
            return "Lightweight Review Mode (Standalone)"
        else:
            return "Full Review Mode with Server Sync"

    def dict(self) -> dict:
        """
        Export the review configuration and its derived operating settings.
        
        Returns:
            dict: Configuration values including server details, reviewer identity,
            review mode, workspace settings, history and cache limits, server mode,
            and synchronization status.
        """
        return {
            "server_url": self.server_url,
            "cache_dir": self.cache_dir,
            "reviewer_name": self.reviewer_name,
            "reviewer_email": self.reviewer_email,
            "mode": self.mode,
            "project_name": self.project_name,
            "workspace_dir": self.workspace_dir,
            "max_history": self.max_history,
            "cache_timeout": self.cache_timeout,
            "server_mode": self.server_mode,
            "auto_sync": self.auto_sync,
        }


# Global config instance
config = ReviewConfig()
