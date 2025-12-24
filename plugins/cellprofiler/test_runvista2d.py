import os

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.setting
import cellprofiler_core.workspace
import numpy
import pytest
from runvista2d import MONAILabelClient, MONAILabelClientException, MONAILabelUtils, RunVISTA2D

IMAGE_NAME = "my_image"
OBJECTS_NAME = "my_objects"
MODEL_NAME = "vista2d"
SERVER_ADDRESS = "http://127.0.0.1:8000"


class MockResponse:
    @staticmethod
    def infer(*args, **kwargs):
        filepath = os.path.abspath(__file__)
        dir = os.path.dirname(filepath)
        image = os.path.join(dir, "resources", "vista2d_test.tiff")
        return image, {}


class MockErrResponse:
    @staticmethod
    def http_multipart(*args, **kwargs):
        return 400, {}, {}, {}


def test_mock_failed():
    x = RunVISTA2D()
    x.y_name.value = OBJECTS_NAME
    x.x_name.value = IMAGE_NAME
    x.server_address.value = SERVER_ADDRESS
    x.model_name.value = MODEL_NAME

    img = numpy.zeros((128, 128, 3))
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.providers.append(cellprofiler_core.image.VanillaImage(IMAGE_NAME, image))
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()

    pytest.MonkeyPatch().setattr(MONAILabelUtils, "http_multipart", MockErrResponse.http_multipart)
    with pytest.raises(MONAILabelClientException):
        x.run(cellprofiler_core.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))


def test_mock_successful():
    x = RunVISTA2D()
    x.y_name.value = OBJECTS_NAME
    x.x_name.value = IMAGE_NAME
    x.server_address.value = SERVER_ADDRESS
    x.model_name.value = MODEL_NAME

    img = numpy.zeros((128, 128, 3))
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.providers.append(cellprofiler_core.image.VanillaImage(IMAGE_NAME, image))
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()

    pytest.MonkeyPatch().setattr(MONAILabelClient, "infer", MockResponse.infer)
    x.run(cellprofiler_core.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
    assert len(object_set.object_names) == 1
    assert OBJECTS_NAME in object_set.object_names
    objects = object_set.get_objects(OBJECTS_NAME)
    segmented = objects.segmented
    assert numpy.all(segmented == 0)
    assert "Image" in measurements.get_object_names()
    assert OBJECTS_NAME in measurements.get_object_names()

    assert f"Count_{OBJECTS_NAME}" in measurements.get_feature_names("Image")
    count = measurements.get_current_measurement("Image", f"Count_{OBJECTS_NAME}")
    assert count == 0
    assert "Location_Center_X" in measurements.get_feature_names(OBJECTS_NAME)
    location_center_x = measurements.get_current_measurement(OBJECTS_NAME, "Location_Center_X")
    assert isinstance(location_center_x, numpy.ndarray)
    assert numpy.product(location_center_x.shape) == 0
    assert "Location_Center_Y" in measurements.get_feature_names(OBJECTS_NAME)
    location_center_y = measurements.get_current_measurement(OBJECTS_NAME, "Location_Center_Y")
    assert isinstance(location_center_y, numpy.ndarray)
    assert numpy.product(location_center_y.shape) == 0
