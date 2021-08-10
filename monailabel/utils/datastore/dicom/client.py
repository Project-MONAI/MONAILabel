import json
import logging
from monailabel.interfaces.datastore import DefaultLabelTag
from typing import Dict

from dicomweb_client.api import DICOMwebClient

from monailabel.utils.datastore.dicom.attributes import (
    ATTRB_MODALITY,
    ATTRB_MONAILABELINFO,
    ATTRB_MONAILABELTAG,
    ATTRB_PATIENTID,
    ATTRB_REFERENCEDSERIESSEQUENCE,
    ATTRB_SERIESINSTANCEUID,
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

            s_meta = self.retrieve_series_metadata(
                study_instance_uid=s_study_id,
                series_instance_uid=s_series_id,
            )
            s_tag = s_meta[ATTRB_MONAILABELTAG]['Value'][0] if s.get(
                ATTRB_MONAILABELTAG) else DefaultLabelTag.FINAL.value
            s_info = json.loads(s_meta[ATTRB_MONAILABELINFO]['Value'][0]) if s.get(ATTRB_MONAILABELINFO) else {}

            # determine if this is a DICOMSEG series
            if s[ATTRB_MODALITY]['Value'][0] == DICOMSEG_MODALITY:

                s_info.update({'object_type': 'label'})

                # add DICOMSEG to datastore
                objects.update({
                    key: DICOMLabelModel(
                        patient_id=s_patient_id,
                        study_id=s_study_id,
                        series_id=s_series_id,
                        tag=s_tag,
                        info=s_info,
                    )
                })

            else:  # this is an original image

                # find all DICOMSEG labels related to this image first
                related_labels_keys = []
                for label in series:
                    label_patient_id = label[ATTRB_PATIENTID]['Value'][0]
                    label_study_id = label[ATTRB_STUDYINSTANCEUID]['Value'][0]
                    label_series_id = label[ATTRB_SERIESINSTANCEUID]['Value'][0]

                    if DICOMSEG_MODALITY in label[ATTRB_MODALITY]['Value'] and s_patient_id == label_patient_id:

                        label_series_meta = self.retrieve_series_metadata(
                            study_instance_uid=label_study_id,
                            series_instance_uid=label_series_id
                        )[0]  # assuming a multiframe DICOMSEG (single-image series)
                        label_referenced_series = []
                        if label_series_meta.get(ATTRB_REFERENCEDSERIESSEQUENCE) and \
                                label_series_meta[ATTRB_REFERENCEDSERIESSEQUENCE]['Value'][0].get(ATTRB_SERIESINSTANCEUID):
                            label_referenced_series.extend(
                                label_series_meta[ATTRB_REFERENCEDSERIESSEQUENCE]['Value'][0]
                                [ATTRB_SERIESINSTANCEUID]['Value'])

                        # to find the related original iage of this label we must look at all instances of a label
                        # in the attribute
                        if s_series_id in label_referenced_series:

                            label_key = generate_key(label_patient_id, label_study_id, label_series_id)
                            related_labels_keys.append(label_key)

                s_info.update({'object_type': 'image'})

                objects.update({
                    key: DICOMImageModel(
                        patient_id=s_patient_id,
                        study_id=s_study_id,
                        series_id=s_series_id,
                        info=s_info,
                        related_labels_keys=related_labels_keys,
                    )
                })

        return objects

    def get_object_url(self, dicom_object: DICOMObjectModel):
        return self._get_series_url('wado', dicom_object.study_id, dicom_object.series_id)

    def get_object(self, dicom_object: DICOMObjectModel):
        return self.retrieve_series(dicom_object.study_id, dicom_object.series_id)

    def push_studies(self):
        pass

    def push_series(self):
        pass
