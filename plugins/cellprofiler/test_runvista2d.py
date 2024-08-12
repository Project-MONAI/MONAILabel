import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.setting
import cellprofiler_core.workspace
import numpy
from runvista2d import RunVISTA2D

IMAGE_NAME = "my_image"
OBJECTS_NAME = "my_objects"
MODEL_NAME = "cell_vista_segmentation"
SERVER_ADDRESS = "http://127.0.0.1:8000"


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
    x.run(cellprofiler_core.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
    assert len(object_set.object_names) == 1
    assert "my_object" in object_set.object_names
    objects = object_set.get_objects("my_object")
    segmented = objects.segmented
    assert numpy.all(segmented == 0)
    assert "Image" in measurements.get_object_names()
    assert "my_object" in measurements.get_object_names()

    assert "Count_my_object" in measurements.get_feature_names("Image")
    count = measurements.get_current_measurement("Image", "Count_my_object")
    assert count == 0
    assert "Location_Center_X" in measurements.get_feature_names("my_object")
    location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
    assert isinstance(location_center_x, numpy.ndarray)
    assert numpy.product(location_center_x.shape) == 0
    assert "Location_Center_Y" in measurements.get_feature_names("my_object")
    location_center_y = measurements.get_current_measurement("my_object", "Location_Center_Y")
    assert isinstance(location_center_y, numpy.ndarray)
    assert numpy.product(location_center_y.shape) == 0
