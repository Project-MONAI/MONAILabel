import itertools
import json
import logging
from typing import Dict

from dicomweb_client.api import DICOMwebClient

from monailabel.utils.datastore.dicom.attributes import (
    ATTRB_MODALITY,
    ATTRB_MONAILABELINFO,
    ATTRB_MONAILABELTAG,
    ATTRB_PATIENTID,
    ATTRB_REFERENCEDIMAGESEQUENCE,
    ATTRB_SERIESINSTANCEUID,
    ATTRB_SOPINSTANCEUID,
    ATTRB_STUDYINSTANCEUID,
    DICOMSEG_MODALITY,
)
from monailabel.utils.datastore.dicom.datamodel import DICOMImageModel, DICOMLabelModel, DICOMObjectModel
from monailabel.utils.datastore.dicom.util import generate_key

logger = logging.getLogger(__name__)


class DICOMWebClient(DICOMwebClient):

    def retrieve_dataset(self) -> Dict[str, DICOMObjectModel]:

        series = self.search_for_series()
        objects: Dict[str, DICOMObjectModel] = {}

        for s in series:
            s_patient_id = s[ATTRB_PATIENTID]['Value'][0]
            s_study_id = s[ATTRB_STUDYINSTANCEUID]['Value'][0]
            s_series_id = s[ATTRB_SERIESINSTANCEUID]['Value'][0]
            key = generate_key(s_patient_id, s_study_id, s_series_id)

            # determine if this is a DICOMSEG series
            if s[ATTRB_MODALITY] == DICOMSEG_MODALITY:

                # add DICOMSEG to datastore
                objects.update({
                    key: DICOMLabelModel(
                        patient_id=s_patient_id,
                        study_id=s_study_id,
                        series_id=s_series_id,
                        tag=s[ATTRB_MONAILABELTAG]['Value'][0] or '',
                        info=json.loads(s[ATTRB_MONAILABELINFO]['Value'][0]) if s.get(ATTRB_MONAILABELINFO) else {},
                    )
                })

            else:  # this is an original image

                # find all DICOMSEG labels related to this image first
                related_labels_keys = []
                for label in series:
                    label_patient_id = label[ATTRB_PATIENTID]['Value'][0]
                    label_study_id = label[ATTRB_STUDYINSTANCEUID]['Value'][0]
                    label_series_id = label[ATTRB_SERIESINSTANCEUID]['Value'][0]

                    if label[ATTRB_MODALITY] == DICOMSEG_MODALITY and s_patient_id == label_patient_id:

                        label_instances = self.search_for_instances(
                            study_instance_uid=label_study_id,
                            series_instance_uid=label_series_id,
                        )
                        label_referenced_instances = [label_instance[ATTRB_REFERENCEDIMAGESEQUENCE]['Value']
                                                      for label_instance in label_instances]
                        label_referenced_instances = set(itertools.chain.from_iterable(label_referenced_instances))

                        original_series_instances = self.search_for_instances(
                            study_instance_uid=s_study_id,
                            series_instance_uid=s_series_id,
                        )
                        original_series_instances = {original_instance[ATTRB_SOPINSTANCEUID]['Value'][0]
                                                     for original_instance in original_series_instances}

                        # to find the related original iage of this label we must look at all instances of a label
                        # in the attribute
                        if label.get(ATTRB_REFERENCEDIMAGESEQUENCE) and \
                           (original_series_instances & label_referenced_instances):

                            label_key = generate_key(label_patient_id, label_study_id, label_series_id)
                            related_labels_keys.append(label_key)

                objects.update({
                    key: DICOMImageModel(
                        patient_id=s_patient_id,
                        study_id=s_study_id,
                        series_id=s_series_id,
                        info=json.loads(s[ATTRB_MONAILABELINFO]['Value'][0]) if s.get(ATTRB_MONAILABELINFO) else {},
                        related_labels_keys=related_labels_keys,
                    )
                })

        return objects

    def push_studies(self):
        pass

    def push_series(self):
        pass
