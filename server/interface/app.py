import json
import os
from abc import abstractmethod


# TODO:: Lets discuss on naming part for most of the interfaces
class MONAIApp(object):
    def __init__(self, name, app_dir, **kwargs):
        self.name = name
        self.app_dir = app_dir

    def meta(self):
        meta_file = os.path.join(self.app_dir, 'meta.json')
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as fc:
                return json.load(fc)
        return {}

    @abstractmethod
    def infer(self, request):
        pass

    @abstractmethod
    def train(self, request):
        pass
