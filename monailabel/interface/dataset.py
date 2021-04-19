import io
import os
import re

import json


class Dataset(object):

    def list_images(self):
        pass

    def list_labels(self):
        pass
 
    def find_objects(self, pattern: str, match_image: bool, match_label: bool):
        pass
  
    def save_label(self, image: str, label_name: str, label: io.BytesIO):
        pass


class LocalDataset(Dataset):

    def __init__(self, dataset_name: str, dataset_path: str, dataset_config: str='dataset.json'):
        self.dataset_path = dataset_path
        self.dataset_config_path = os.path.join(dataset_path, dataset_config)
        self.dataset_config = None
        
        # check if dataset configuration file exists
        if os.path.exists(self.dataset_config_path):
            with open(self.dataset_config_path) as f:
                dataset_config = json.load(f)
        else:
            files = LocalDataset._list_files(dataset_path)
            self.dataset_config = {
                    'name': dataset_name,
                    'objects': [{
                        'image': file,
                        'labels': [],
                    } for file in files],
                }

            self._update_dataset()

    def list_images(self):
        return [obj['image'] for obj in self.dataset_config['objects']]

    def list_labels(self):
        return [obj['labels'] for obj in self.dataset_config['objects']]

    def find_objects(self, pattern: str, match_image: bool=True, match_label: bool=True):
        p = re.compile(pattern)
        matching_objects = []
        for obj in self.dataset_config['objects']:
            if p.match(obj['image']) and match_image:
                matching_objects.append(obj)
            if any([p.match(l) for l in obj['labels']]) and match_label:
                matching_objects.append(obj)
        return matching_objects
    
    def save_label(self, image: str, label_name: str, label: io.BytesIO):
        
        for obj in self.dataset_config['objects']:
        
            if image == self.dataset_config['image']:
                label_path = self.dataset_config['image']
                self.dataset_config['labels'].append(os.path.join(os.path.basename(self.dataset_config['image']), label_name))

                with open(self.dataset_config['label'][-1], 'wb') as f:
                    label.seek(0)
                    f.write(label.getbuffer())
        
        self._update_dataset()
    
    @staticmethod
    def _list_files(path: str):
        relative_file_paths = []
        for root, dirs, files in os.walk(path):
            base_dir = root.strip(path)
            relative_file_paths.extend([os.path.join(base_dir, file) for file in files])
        return relative_file_paths

    def _update_dataset(self):
        with open(self.dataset_config_path, 'w') as f:
                json.dump(self.dataset_config, f)

if __name__ == "__main__":
    ds = LocalDataset('Task09', '/raid/datasets/Task09_Spleen/imagesTs')