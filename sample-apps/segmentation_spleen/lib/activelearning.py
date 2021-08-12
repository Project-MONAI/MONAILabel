import logging

from monailabel.interfaces import Datastore
from monailabel.interfaces.tasks import Strategy

logger = logging.getLogger(__name__)


class MyStrategy(Strategy):
    """
    Consider implementing a first strategy for active learning
    """

    def __init__(self):
        super().__init__("Get First Sample")

    def __call__(self, request, datastore: Datastore):
        images = datastore.get_unlabeled_images()
        if not len(images):
            return None

        images.sort()
        image = images[0]

        logger.info(f"First: Selected Image: {image}")
        return image


class VarianceStrategy(Strategy):
    """
    Using variance of the MC samples generated
    """

    '''
    #TODO Sample Code
    def variance_3d_volume(vol_input):
        # The input is assumed with repetitions, channels and then volumetric data

        vol_input = vol_input.astype(dtype='float32')
        dims = vol_input.shape

        # Threshold values less than or equal to zero
        threshold = 0.0005
        vol_input[vol_input<=0] = threshold

        vari = np.nanvar(vol_input, axis=0)
        variance = np.sum(vari, axis=0)

        return variance

    '''

    '''
    def entropy_3d_volume(vol_input):
        # The input is assumed with repetitions, channels and then volumetric data
        vol_input = vol_input.astype(dtype='float32')
        dims = vol_input.shape
        reps = dims[0]
        entropy = np.zeros(dims[2:], dtype='float32')

        # Threshold values less than or equal to zero
        threshold = 0.0005
        vol_input[vol_input <= 0] = threshold

        # Looping across channels as each channel is a class
        for channel in range(dims[1]):
            t_vol = np.squeeze(vol_input[:, channel, :, :, :])
            t_sum = np.sum(t_vol, axis=0)
            t_avg = np.divide(t_sum, reps)
            t_log = np.log(t_avg)
            t_entropy = -np.multiply(t_avg, t_log)
            entropy = entropy + t_entropy
        return entropy

    '''

    # TODO Questions to be answered
    # 1.) How does the Strategy request the model to give it MC samples?
    # 2.) The models need to be modified to ensure we have dropout (Creating a new issue for this)
    # 3.) How is the code design going to be, do we need a new 'Class' per strategy or we can have multiple strategies in a single one

    def __init__(self):
        super().__init__("Get First Sample")

    def __call__(self, request, datastore: Datastore):
        images = datastore.get_unlabeled_images()
        if not len(images):
            return None

        images.sort()
        image = images[0]

        logger.info(f"First: Selected Image: {image}")
        return image
