from collections import OrderedDict
import numpy as np
import nibabel as nib

class ExperimentPlanner(object):

    def __init__(self, datastore):
        self.plans = OrderedDict()
        self.target_spacing_percentile = 50
        self.datastore = datastore

    def get_target_spacing(self):
        spacings = []
        for n in self.datastore:
            img = nib.load(n)
            affine = img.affine
            spacing = nib.aff2axcodes(affine)
            spacing.append(spacing)
        spacings = np.array(spacings)
        target = np.percentile(np.vstack(spacings), self.target_spacing_percentile, 0)
        return target
