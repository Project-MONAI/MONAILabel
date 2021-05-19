from typing import List

import requests
from watchdog.events import PatternMatchingEventHandler


class DataStoreHandler(PatternMatchingEventHandler):
    def __init__(
        self,
        api_str: str,
        studies: str,
        host_url: str = "localhost",
        host_port: str = "8000",
        refresh_endpoint: str = "/datastore",
        patterns: List[str] = ["**/*.nii.gz", "**/*.nii", "**/*.npy", "**/*.npz"],
    ):
        self.refresh_url = f"http://{host_url}:{host_port}{api_str}{refresh_endpoint}"
        super(DataStoreHandler, self).__init__(patterns=[f"{studies}/{ext}" for ext in patterns])

    def on_any_event(self, event):
        self._refresh()

    def _refresh(self):
        dl = requests.get(self.refresh_url)
