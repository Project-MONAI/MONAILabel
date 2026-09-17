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
        Initialize reviewer configuration.

        Parameters:
        - server_url: MONAI Label server URL (e.g., http://localhost:8000)
                     If None, runs in standalone mode using local cache
        - cache_dir: Directory for cache storage
        - reviewer_name: Name of the reviewer
        - reviewer_email: Email of the reviewer
        - mode: Operating mode - "review" (requires server) or "standalone" (local cache only)
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
        """Return the server mode description."""
        if self.review_only:
            return "Lightweight Review Mode (Standalone)"
        else:
            return "Full Review Mode with Server Sync"

    def dict(self) -> dict:
        """Return configuration as dictionary."""
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


# Lazily created to avoid filesystem side-effects at import time.
_config: Optional[ReviewConfig] = None


def get_config() -> ReviewConfig:
    global _config
    if _config is None:
        _config = ReviewConfig()
    return _config
