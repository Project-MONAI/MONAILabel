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

import datetime
import logging
import os
import re
import tempfile
from typing import Dict, List

import qt
import requests
import SampleData
import slicer
from MONAILabelReviewerLib.ImageData import ImageData
from MONAILabelReviewerLib.ImageDataController import ImageDataController, ImageDataStatistics
from MONAILabelReviewerLib.MONAILabelReviewerEnum import Label, Level, SegStatus
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


class MONAILabelReviewer(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MONAILabel Reviewer"
        self.parent.categories = ["Active Learning"]
        self.parent.dependencies = []
        self.parent.contributors = ["Minh Duc, Do (rAIdiance)"]
        self.parent.helpText = """
This module provides the user to review on segmentations on X-Ray-dicom images.
See more information in <a href="...">module documentation</a>.
"""
        self.parent.acknowledgementText = """
Developed by rAiDiance, and  funded by Berlin Institute of Health (BIH).
"""


class MONAILabelReviewerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        # Color set
        self.colorGreenPressedButton = "background-color : rgb(0, 250, 146)"
        self.colorDarkGrayButton = "background-color : rgb(169, 169, 169)"
        self.colorGreenButtonAfterSuccessfulLoad = "background-color : rgb(0, 144, 81)"
        self.colorGreenEasyButton = "background-color : rgb(0, 250, 146)"
        self.colorYellowMediumButton = "background-color : rgba(255, 251, 0, 179)"
        self.colorRedHardButton = "background-color : rgba(255, 38, 0, 179)"
        self.colorLightGreenButton = "background-color : rgb(115, 250, 121)"
        self.colorRedReviewerModeButton = "background-color : rgb(255, 126, 121)"
        self.colorBlueBasicModeButton = "background-color : rgb(118, 214, 255)"
        self.colorRed = "color: red"
        self.colorGreen = "color: green"
        self.colorLightYellow = "background-color : rgb(255,255,153)"

        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        self.STATUS = SegStatus()
        self.LEVEL = Level()
        self.LABEL = Label()

        self.selectedReviewer: str = ""
        self.selectedClientId: str = ""
        self.currentImageId: str = ""
        self.currentLabelVersion: str = ""
        self.listImageData: List[ImageData] = None
        self.imageCounter: int = 0
        self.currentImageData: ImageData = None
        self.idToimageData: Dict[str, ImageData] = None

        # Meta Information
        self.finalStatus: str = ""
        self.finalLevel: str = ""
        self.finalComment: str = ""
        self.tmpdir = ""

        self.reviewersModeIsActive = False
        self.isSelectableByLabelVersion = False

        self.mapFiltersToBool: Dict[str, bool] = {
            "segmented": False,
            "notSegemented": False,
            "approved": False,
            "flagged": False,
        }

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MONAILabelReviewer.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MONAILabelReviewerLogic()

        # set segmentator editor
        self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        self.addSegmentator()
        self.setLightVersion()

        self.ui.verticalLayout_10.addWidget(self.segmentEditorWidget)
        self.loadServerSelection()

        #  Section: Widget Elements
        self.ui.btn_connect_monai.clicked.connect(self.init_dicom_stream)
        self.ui.btn_load.clicked.connect(self.loadImageData)

        self.ui.btn_approved.clicked.connect(self.approveSegmentation)
        self.ui.btn_mark_revision.clicked.connect(self.flagSegmentation)

        self.ui.btn_next.clicked.connect(self.getNextSegmentation)
        self.ui.btn_previous.clicked.connect(self.getPreviousSegmenation)

        self.ui.btn_easy.clicked.connect(self.setEasy)
        self.ui.btn_medium.clicked.connect(self.setMedium)
        self.ui.btn_hard.clicked.connect(self.setHard)

        self.ui.btn_search.clicked.connect(self.search)
        self.ui.btn_search_annotator_reviewer.clicked.connect(self.searchByAnnotatorReviewer)
        self.ui.btn_search_level.clicked.connect(self.searchByLevel)

        self.ui.checkBox_search_approved.clicked.connect(self.checkedAppprovedSearch)
        self.ui.checkBox_search_flagged.clicked.connect(self.checkedFlaggedSearch)

        self.ui.btn_show_image.clicked.connect(self.showSearchedImage)

        self.ui.checkBox_flagged.clicked.connect(self.checkedFlagged)
        self.ui.checkBox_approved.clicked.connect(self.checkApproved)
        self.ui.checkBox_not_segmented.clicked.connect(self.checkNotSegmented)
        self.ui.checkBox_segmented.clicked.connect(self.checkSegmented)

        self.ui.btn_basic_mode.clicked.connect(self.setLightVersion)
        self.ui.btn_reviewers_mode.clicked.connect(self.setReviewerVersion)
        self.ui.comboBox_clients.currentIndexChanged.connect(self.index_changed)
        self.ui.comboBox_reviewers.currentIndexChanged.connect(self.indexReviewerchanged)

        self.ui.comboBox_label_version.currentIndexChanged.connect(self.indexLabelVersionChanged)
        self.ui.btn_save_new_version.clicked.connect(self.setSaveAsNewVersion)
        self.ui.btn_overwrite_version.clicked.connect(self.setOverwriteCurrentVersion)
        self.ui.btn_update_version.clicked.connect(self.updateAfterEditingSegmentation)
        self.ui.btn_edit_label.clicked.connect(self.displayEditorTools)
        self.ui.btn_delete_version.clicked.connect(self.setDeleteVersion)

    def getCurrentTime(self):
        return datetime.datetime.now()

    def getCurrentMetaStatus(self) -> str:
        if not self.finalStatus:
            return ""
        return self.finalStatus

    def setCurrentMetaStatus(self, status=""):
        self.finalStatus = status

    def getCurrentMetaLevel(self) -> str:
        if not self.finalLevel:
            return ""
        return self.finalLevel

    def setCurrentMetaLevel(self, level=""):
        self.finalLevel = level

    def getCurrentComment(self) -> str:
        comment = self.ui.plainText_comment.toPlainText()
        if not comment:
            return ""
        return comment

    def setCurrentComment(self, comment=""):
        self.finalComment = comment

    def getSelectedReviewer(self) -> str:
        selectedReviewer = self.ui.comboBox_reviewers.currentText
        if not selectedReviewer:
            return ""
        return selectedReviewer

    def getSelectedClientFromComboBox(self) -> str:
        selectedClient = self.ui.comboBox_clients.currentText
        if not selectedClient:
            return ""
        return selectedClient

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def indexReviewerchanged(self, index):
        logging.info(f"{self.getCurrentTime()}: Selected reviewer: '{self.ui.comboBox_reviewers.currentText}'")
        self.selectedReviewer = self.ui.comboBox_reviewers.currentText

    def index_changed(self, index):
        self.loadImageData()

    def indexLabelVersionChanged(self, index):
        logging.warn(
            f"{self.getCurrentTime()}: Selected labal version: '{self.getCurrentLabelVersionFromComboBox()}', is enabled '{self.isSelectableByLabelVersion}'"
        )
        self.displayAdditionalMetaIfEdited(self.getCurrentLabelVersionFromComboBox())
        if self.isSelectableByLabelVersion:
            self.disableDifficultyButtons(tag=self.getCurrentLabelVersion())
            self.loadNextImage(imageData=self.currentImageData, tag=self.getCurrentLabelVersion())

    def getCurrentLabelVersionFromComboBox(self) -> str:
        labelString = self.ui.comboBox_label_version.currentText
        return self.parseSelectedVersionFromComboBox(labelString)

    def getCurrentLabelVersion(self) -> str:
        label = self.getCurrentLabelVersionFromComboBox()
        if label == "":
            label = self.LABEL.FINAL
        return label

    def disableDifficultyButtons(self, tag: str):
        if self.LABEL.VERSION in tag:
            self.ui.btn_easy.hide()
            self.ui.btn_medium.hide()
            self.ui.btn_hard.hide()
        else:
            self.ui.btn_easy.show()
            self.ui.btn_medium.show()
            self.ui.btn_hard.show()

    def setSaveAsNewVersion(self) -> bool:
        setToSave = bool(self.ui.btn_save_new_version.isChecked())
        self.ui.btn_update_version.enabled = True
        if setToSave:
            self.ui.btn_overwrite_version.setChecked(False)
            self.ui.btn_overwrite_version.setStyleSheet(self.colorDarkGrayButton)

            self.ui.btn_delete_version.setChecked(False)
            self.ui.btn_delete_version.setStyleSheet(self.colorDarkGrayButton)

            self.ui.btn_save_new_version.setStyleSheet(self.colorGreenPressedButton)
            self.ui.btn_update_version.setText("Confirm: Saving")
        return setToSave

    def setOverwriteCurrentVersion(self) -> bool:
        setToOverwrite = bool(self.ui.btn_overwrite_version.isChecked())
        self.ui.btn_update_version.enabled = True
        if setToOverwrite:

            self.ui.btn_save_new_version.setChecked(False)
            self.ui.btn_save_new_version.setStyleSheet(self.colorDarkGrayButton)

            self.ui.btn_delete_version.setChecked(False)
            self.ui.btn_delete_version.setStyleSheet(self.colorDarkGrayButton)

            self.ui.btn_overwrite_version.setStyleSheet(self.colorGreenPressedButton)
            self.ui.btn_update_version.setText("Confirm: Overwriting")

        return setToOverwrite

    def setDeleteVersion(self) -> bool:
        setToDelete = bool(self.ui.btn_delete_version.isChecked())
        self.ui.btn_update_version.enabled = True
        if setToDelete:
            self.ui.btn_save_new_version.setChecked(False)
            self.ui.btn_save_new_version.setStyleSheet(self.colorDarkGrayButton)

            self.ui.btn_overwrite_version.setChecked(False)
            self.ui.btn_overwrite_version.setStyleSheet(self.colorDarkGrayButton)

            self.ui.btn_delete_version.setStyleSheet(self.colorGreenPressedButton)
            self.ui.btn_update_version.setText("Confirm: Deletion")

        return setToDelete

    def setButtonColorReviewerOrBasicMode(self, isReviewerMode: bool):
        if isReviewerMode:
            self.ui.btn_reviewers_mode.setStyleSheet(self.colorRedReviewerModeButton)
            self.ui.btn_basic_mode.setStyleSheet(self.colorDarkGrayButton)
        else:
            self.ui.btn_reviewers_mode.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_basic_mode.setStyleSheet(self.colorBlueBasicModeButton)

    def setReviewerVersion(self):
        self.reviewersModeIsActive = True
        self.setButtonColorReviewerOrBasicMode(isReviewerMode=self.reviewersModeIsActive)
        # section: Server
        # Reviewer Field
        self.ui.label_20.show()
        self.ui.comboBox_reviewers.show()

        # Approved bar
        self.ui.label_17.show()
        self.ui.progressBar_approved_total.show()
        self.ui.label_idx_appr_image.show()

        # section: Data set explorer

        # Approved bar
        self.ui.label_10.show()
        self.ui.progressBar_approved_client.show()
        self.ui.label_idx_appr_image_client.show()

        # filter option
        self.ui.label_6.show()
        self.ui.checkBox_not_segmented.show()
        self.ui.checkBox_flagged.show()
        self.ui.checkBox_segmented.show()
        self.ui.checkBox_approved.show()

        # section: Data evaluation
        self.ui.btn_easy.show()
        self.ui.btn_medium.show()
        self.ui.btn_hard.show()
        self.ui.label_level_difficulty.show()
        self.ui.btn_mark_revision.show()
        self.ui.btn_approved.show()

        # imag information
        self.ui.label_14.show()
        self.ui.lineEdit_status.show()
        self.ui.label_16.show()
        self.ui.lineEdit_level.show()
        self.ui.plainText_comment.show()
        if self.ui.btn_basic_mode.isChecked():
            self.ui.btn_basic_mode.setChecked(False)

        self.collapseAllSecions()

        self.activateSegmentatorEditor(activated=False)
        self.hideEditingSelectionOption(isHidden=False)

    # Section:  Light version Option
    def setLightVersion(self):
        self.reviewersModeIsActive = False
        self.setButtonColorReviewerOrBasicMode(isReviewerMode=self.reviewersModeIsActive)
        # section: Server
        # Reviewer Field
        self.ui.label_20.hide()
        self.ui.comboBox_reviewers.hide()

        # Approved bar
        self.ui.label_17.hide()
        self.ui.progressBar_approved_total.hide()
        self.ui.label_idx_appr_image.hide()

        # section: Data set explorer
        # Approved bar
        self.ui.label_10.hide()
        self.ui.progressBar_approved_client.hide()
        self.ui.label_idx_appr_image_client.hide()

        # filter option
        self.ui.label_6.hide()
        self.ui.checkBox_not_segmented.hide()
        self.ui.checkBox_flagged.hide()
        self.ui.checkBox_segmented.hide()
        self.ui.checkBox_approved.hide()

        # section: Data evaluation
        self.ui.btn_easy.hide()
        self.ui.btn_medium.hide()
        self.ui.btn_hard.hide()
        self.ui.label_level_difficulty.hide()
        self.ui.btn_mark_revision.hide()
        self.ui.btn_approved.hide()

        # imag information
        self.ui.label_14.hide()
        self.ui.lineEdit_status.hide()
        self.ui.label_16.hide()
        self.ui.lineEdit_level.hide()
        self.ui.plainText_comment.hide()
        if self.ui.btn_reviewers_mode.isChecked():
            self.ui.btn_reviewers_mode.setChecked(False)

        self.collapseAllSecions()

        self.activateSegmentatorEditor(activated=False)
        self.hideEditingSelectionOption(isHidden=True)

    def cleanCache(self):
        self.logic = MONAILabelReviewerLogic()
        self.selectedReviewer = ""
        self.selectedClientId = ""

        self.listImageData = None
        self.imageCounter = 0
        self.currentImageData = None
        self.idToimageData = None

        # Meta Information
        self.setCurrentMetaStatus(status="")
        self.setCurrentMetaLevel(level="")
        self.setCurrentComment(comment="")

        logging.info(f"{self.getCurrentTime()}: Cache is cleaned")

    # Section: Server
    def loadServerSelection(self):
        settings = qt.QSettings()
        serverUrlHistory = settings.value("MONAILabel/serverUrlHistory")

        self.ui.comboBox_server_url.clear()
        self.ui.comboBox_server_url.addItems(serverUrlHistory.split(";"))

    def init_dicom_stream(self):
        """
        initiates connection to monai server
        Default: client listens on "http://127.0.0.1:8000"
        """
        # Check Connection
        self.cleanCache()
        serverUrl: str = self.ui.comboBox_server_url.currentText
        isConnected: bool = self.logic.connectToMonaiServer(serverUrl)
        if not isConnected:
            warningMessage = f"Connection to server failed \ndue to invalid ip '{serverUrl}'"
            slicer.util.warningDisplay(warningMessage)
            return
        self.ui.btn_connect_monai.setStyleSheet(self.colorGreenButtonAfterSuccessfulLoad)
        self.processDataStoreRecords()
        self.initUI()

    def collapseAllSecions(self):
        self.ui.collapsibleButton_search_image.enabled = False
        self.ui.collapsibleButton_dicom_stream.enabled = False
        self.ui.collapsibleButton_dicom_evaluation.enabled = False

        self.ui.collapsibleButton_search_image.collapsed = True
        self.ui.collapsibleButton_dicom_stream.collapsed = True
        self.ui.collapsibleButton_dicom_evaluation.collapsed = True

    def initUI(self):
        self.selectedReviewer = self.ui.comboBox_reviewers.currentText
        if self.reviewersModeIsActive and self.selectedReviewer == "":
            warningMessage = "Missing reviewer's name.\nPlease enter your id or name in the reviewer's field!"
            slicer.util.warningDisplay(warningMessage)
            return
        self.ui.collapsibleButton_search_image.enabled = True
        self.ui.collapsibleButton_dicom_stream.enabled = True

        # set Segmentation progress bar
        self.setProgessBar()

        # fill combobox
        self.fillComboBoxes()

        # set up buttons
        self.setButtons()

        self.selectedClientId = ""

    def setButtons(self):
        self.ui.btn_approved.setCheckable(True)
        self.ui.btn_mark_revision.setCheckable(True)
        self.ui.btn_easy.setCheckable(True)
        self.ui.btn_medium.setCheckable(True)
        self.ui.btn_hard.setCheckable(True)
        self.ui.btn_reviewers_mode.setCheckable(True)
        self.ui.btn_basic_mode.setCheckable(True)

        self.ui.btn_edit_label.setCheckable(True)
        self.ui.btn_save_new_version.setCheckable(True)
        self.ui.btn_overwrite_version.setCheckable(True)
        self.ui.btn_delete_version.setCheckable(True)
        self.ui.btn_update_version.setCheckable(True)

        self.ui.btn_delete_version.hide()
        self.ui.btn_save_new_version.hide()
        self.ui.btn_overwrite_version.hide()
        self.ui.btn_update_version.hide()
        self.ui.btn_update_version.enabled = False

        self.ui.btn_show_image.enabled = False

    def setProgessBar(self):
        statistics = self.logic.getStatistics()

        self.ui.progressBar_segmentation.setProperty("value", statistics.getSegmentationProgress())
        self.ui.label_idx_seg_image.setText(statistics.getIdxTotalSegmented())
        self.ui.label_idx_appr_image.setText(statistics.getIdxTotalApproved())
        self.ui.progressBar_approved_total.setProperty("value", statistics.getProgressPercentage())

    def fillComboBoxes(self):
        # clients
        clientIds = self.logic.getClientIds()

        self.ui.comboBox_clients.clear()
        self.ui.comboBox_clients.addItem("All")
        for clientId in clientIds:
            self.ui.comboBox_clients.addItem(str(clientId))

        # combobox in search section
        self.ui.comboBox_search_annotator.clear()
        self.ui.comboBox_search_annotator.addItem("All")
        for clientId in clientIds:
            self.ui.comboBox_search_annotator.addItem(str(clientId))

        # reviewers
        self.ui.comboBox_reviewers.clear()
        reviewers = self.logic.getReviewers()
        self.ui.comboBox_reviewers.addItem(self.selectedReviewer)

        for reviewer in reviewers:
            if reviewer == self.selectedReviewer:
                continue
            self.ui.comboBox_reviewers.addItem(str(reviewer))
        self.ui.comboBox_reviewers.setCurrentText(self.selectedReviewer)

        # combobox in search section
        self.ui.comboBox_search_reviewer.clear()
        self.ui.comboBox_search_reviewer.addItem("All")
        for reviewer in reviewers:
            self.ui.comboBox_search_reviewer.addItem(str(reviewer))

    def cleanDicomStreamSection(self):

        self.setCurrentMetaStatus(status="")
        self.setCurrentMetaLevel(level="")
        self.setCurrentComment(comment="")

        self.selectedClientId = None
        self.imageCounter = 0
        self.currentImageData = None
        self.idToimageData = None
        self.listImageData = None

        self.cleanProgressBarDicomStreamSection()
        self.cleanCheckBoxes()
        self.resetHorizontalSlider()

    # Section: Loading images
    def loadImageData(self):
        if (self.selectedClientId == self.getSelectedClientFromComboBox()) and (self.isDifferentFilter() is False):
            return
        self.imageCounter = 0

        self.cleanSearchSection()
        # select segmentator: ALL
        self.selectedClientId = self.getSelectedClientFromComboBox()
        if self.selectedClientId == "All":
            self.listImageData = self.loadImageDataWithFilter(selectedClientId="")
            self.ui.checkBox_segmented.setEnabled(True)
            self.ui.checkBox_not_segmented.setEnabled(True)
            self.setProgressBarOfAll()

        # select segmentator: client was selected
        if self.selectedClientId != "All":
            self.listImageData = self.loadImageDataWithFilter(selectedClientId=self.selectedClientId)
            self.setCheckBoxesClient()
            self.setProgressBarOfClient(self.selectedClientId)

        logging.info(
            "{}: Successfully loaded Image data [total = {}, category = '{}']".format(
                self.getCurrentTime(), len(self.listImageData), self.selectedClientId
            )
        )

        if len(self.listImageData) > 0:
            self.currentImageData = self.listImageData[self.imageCounter]
            self.loadNextImage(self.currentImageData)

        self.ui.collapsibleButton_dicom_evaluation.enabled = True
        self.ui.collapsibleButton_dicom_evaluation.collapsed = False
        self.setHorizontalSlider(len(self.listImageData))
        self.collectFilters()
        self.setLoadButtonColor(reload=False)

    def loadImageDataWithFilter(self, selectedClientId: str) -> list:
        isApproved = bool(self.ui.checkBox_approved.isChecked())
        isFlagged = bool(self.ui.checkBox_flagged.isChecked())
        isNotSegmented = bool(self.ui.checkBox_not_segmented.isChecked())
        segmented = bool(self.ui.checkBox_segmented.isChecked())
        logging.info(
            "{}: Selected filters: segmented= {} | isNotSegmented= {} | isApproved= {} | isFlagged= {}".format(
                self.getCurrentTime(), segmented, isNotSegmented, isApproved, isFlagged
            )
        )
        if selectedClientId == "":
            return self.logic.getAllImageData(segmented, isNotSegmented, isApproved, isFlagged)
        return self.logic.getImageDataByClientId(selectedClientId, isApproved, isFlagged)

    def setProgressBarOfAll(self):
        statistics: ImageDataStatistics = self.logic.getStatistics()
        # Progress bar: Segmented/TotalImage
        self.ui.progressBar_segmented_client.setProperty("value", statistics.getSegmentationProgressAllPercentage())
        self.ui.label_idx_seg_image_client.setText(statistics.getIdxTotalSegmented())
        # Progress bar: approvalCount/TotalImage
        self.ui.progressBar_approved_client.setProperty("value", statistics.getApprovalProgressPercentage())
        self.ui.label_idx_appr_image_client.setText(statistics.getIdxTotalApproved())

    def cleanProgressBarDicomStreamSection(self):
        self.ui.progressBar_segmented_client.setProperty("value", 0)
        self.ui.progressBar_approved_client.setProperty("value", 0)
        self.ui.label_idx_seg_image_client.setText("x/y")
        self.ui.label_idx_appr_image_client.setText("x/y")

    def setLoadButtonColor(self, reload: bool):
        if reload:  # reload required
            self.ui.btn_load.setStyleSheet(self.colorDarkGrayButton)
            return
        self.ui.btn_load.setStyleSheet(self.colorGreenButtonAfterSuccessfulLoad)

    def setProgressBarOfClient(self, selectedClientId: str):
        percentageApprovedOfClient, idxApprovedOfClient = self.logic.getPercentageApproved(selectedClientId)
        self.ui.progressBar_approved_client.setProperty("value", percentageApprovedOfClient)
        self.ui.label_idx_appr_image_client.setText(idxApprovedOfClient)

        percentageSemgmentedByClient, idxSegmentedByClient = self.logic.getPercentageSemgmentedByClient(
            selectedClientId
        )
        self.ui.progressBar_segmented_client.setProperty("value", percentageSemgmentedByClient)
        self.ui.label_idx_seg_image_client.setText(idxSegmentedByClient)

    def setHorizontalSlider(self, loadesImageCount: int):
        self.ui.horizontalSlider_image_idx.setMinimum(0)
        self.ui.horizontalSlider_image_idx.setMaximum(loadesImageCount - 1)
        idxImage = f"Image: {self.imageCounter + 1}/{len(self.listImageData)}"
        self.ui.label_idx_image.setText(idxImage)

    def updateHorizontalSlider(self):
        self.ui.horizontalSlider_image_idx.setValue(self.imageCounter)
        idxImage = f"Image: {self.imageCounter + 1}/{len(self.listImageData)}"
        self.ui.label_idx_image.setText(idxImage)

    def resetHorizontalSlider(self):
        self.ui.horizontalSlider_image_idx.setValue(1)
        self.ui.label_idx_image.setText("Image:")

    # Section: Filter
    def collectFilters(self):
        self.mapFiltersToBool["segmented"] = self.ui.checkBox_segmented.isChecked()
        self.mapFiltersToBool["notSegemented"] = self.ui.checkBox_not_segmented.isChecked()
        self.mapFiltersToBool["approved"] = self.ui.checkBox_approved.isChecked()
        self.mapFiltersToBool["flagged"] = self.ui.checkBox_flagged.isChecked()

    def isDifferentFilter(self) -> bool:
        if self.mapFiltersToBool["segmented"] != self.ui.checkBox_segmented.isChecked():
            return True
        if self.mapFiltersToBool["notSegemented"] != self.ui.checkBox_not_segmented.isChecked():
            return True
        if self.mapFiltersToBool["approved"] != self.ui.checkBox_approved.isChecked():
            return True
        if self.mapFiltersToBool["flagged"] != self.ui.checkBox_flagged.isChecked():
            return True
        return False

    # CheckBox: clean
    def cleanCheckBoxes(self):
        self.ui.checkBox_segmented.setChecked(False)
        self.ui.checkBox_not_segmented.setChecked(False)
        self.ui.checkBox_flagged.setChecked(False)
        self.ui.checkBox_approved.setChecked(False)

    # CheckBox: flagged
    def setCheckBoxesClient(self):
        self.setLoadButtonColor(reload=True)
        self.ui.checkBox_not_segmented.setEnabled(False)
        self.ui.checkBox_segmented.setChecked(True)
        self.ui.checkBox_segmented.setEnabled(False)

    # CheckBox: flagged
    def checkedFlagged(self):
        self.setLoadButtonColor(reload=True)
        self.ui.checkBox_segmented.setChecked(True)
        if self.ui.checkBox_approved.isChecked():
            self.ui.checkBox_approved.setChecked(False)
        if self.ui.checkBox_not_segmented.isChecked():
            self.ui.checkBox_not_segmented.setChecked(False)

    # CheckBox: approved
    def checkApproved(self):
        self.setLoadButtonColor(reload=True)
        self.ui.checkBox_segmented.setChecked(True)
        if self.ui.checkBox_flagged.isChecked():
            self.ui.checkBox_flagged.setChecked(False)
        if self.ui.checkBox_not_segmented.isChecked():
            self.ui.checkBox_not_segmented.setChecked(False)

    # CheckBox: NOT segmented
    def checkNotSegmented(self):
        self.setLoadButtonColor(reload=True)
        if self.ui.checkBox_approved.isChecked():
            self.ui.checkBox_approved.setChecked(False)
        if self.ui.checkBox_flagged.isChecked():
            self.ui.checkBox_flagged.setChecked(False)
        if self.ui.checkBox_segmented.isChecked():
            self.ui.checkBox_segmented.setChecked(False)

    # CheckBox: segmented
    def checkSegmented(self):
        self.setLoadButtonColor(reload=True)
        if self.ui.checkBox_segmented.isChecked() is False:
            self.ui.checkBox_approved.setChecked(False)
            self.ui.checkBox_flagged.setChecked(False)
            return

        if self.ui.checkBox_not_segmented.isChecked():
            self.ui.checkBox_not_segmented.setChecked(False)

    # Section: Search Image
    def cleanSearchSection(self):
        self.ui.tableWidge_imageMeta.setRowCount(0)
        self.ui.tableWidge_imageMeta.clearContents()
        self.ui.textEdit_search.clear()

    def search(self):
        """
        After triggering search button, load images and segmentation by input ids
        """
        self.cleanDicomStreamSection()

        if self.ui.textEdit_search.toPlainText() == "":
            logging.info(f"{self.getCurrentTime()}: Search input field is empty")
            return

        idsStr = self.ui.textEdit_search.toPlainText()
        idList = self.getIdsFromString(idsStr)

        self.idToimageData = self.logic.getMultImageDataByIds(idList)
        self.listImageData = [*self.idToimageData.values()]

        foundIdList = [imageData.getName() for imageData in self.listImageData]
        notFoundIdList = [id for id in idList if (id not in foundIdList)]
        self.loadSearchImageMetaInTable(self.listImageData, notFoundIdList)

        self.ui.collapsibleButton_dicom_evaluation.enabled = True
        self.setHorizontalSlider(len(foundIdList))
        if len(foundIdList) > 0:
            self.setSearchResultMessage(numOfFound=len(foundIdList))
            self.loadFirstImage()
        else:
            self.setSearchResultMessage(numOfFound=0)

    def searchByAnnotatorReviewer(self):
        selectedAnnotator: str = self.ui.comboBox_search_annotator.currentText
        selectedReviewer: str = self.ui.comboBox_search_reviewer.currentText
        isApproved: bool = bool(self.ui.checkBox_search_approved.isChecked())
        isFlagged: bool = bool(self.ui.checkBox_search_flagged.isChecked())
        logging.warn(
            f"{self.getCurrentTime()}: Search by annontator: '{selectedAnnotator}' | reviewer: '{selectedReviewer}' | isApproved: '{isApproved}' | isFlagged: '{isFlagged}'"
        )

        self.idToimageData = self.logic.searchByAnnotatorReviewer(
            selectedAnnotator, selectedReviewer, isApproved, isFlagged
        )
        self.listImageData = [*self.idToimageData.values()]

        self.loadSearchImageMetaInTable(self.listImageData, [])
        if len(self.listImageData) > 0:
            self.ui.collapsibleButton_dicom_evaluation.enabled = True
            self.setSearchResultMessage(numOfFound=len(self.idToimageData))
            self.setHorizontalSlider(len(self.idToimageData))
            self.loadFirstImage()
        else:
            self.setSearchResultMessage(numOfFound=0)

    def searchByLevel(self):
        isEasy: bool = bool(self.ui.checkBox_search_easy.isChecked())
        isMedium: bool = bool(self.ui.checkBox_search_medium.isChecked())
        isHard: bool = bool(self.ui.checkBox_search_hard.isChecked())

        self.idToimageData = self.logic.searchByLevel(isEasy, isMedium, isHard)
        self.listImageData = [*self.idToimageData.values()]

        self.loadSearchImageMetaInTable(self.listImageData, [])
        if len(self.listImageData) > 0:
            self.ui.collapsibleButton_dicom_evaluation.enabled = True
            self.setSearchResultMessage(numOfFound=len(self.idToimageData))
            self.setHorizontalSlider(len(self.idToimageData))
            self.loadFirstImage()
        else:
            self.setSearchResultMessage(numOfFound=0)

    def setSearchResultMessage(self, numOfFound: int):
        if numOfFound == 0:
            self.ui.label_search_result.setText("Result: No images found.")
            self.ui.label_search_result.setStyleSheet(self.colorRed)
        else:
            resultMessage = f"Result: {numOfFound} images found."
            self.ui.label_search_result.setText(resultMessage)
            self.ui.label_search_result.setStyleSheet(self.colorGreen)

    def checkedAppprovedSearch(self):
        isFlagged = bool(self.ui.checkBox_search_flagged.isChecked())
        if isFlagged:
            self.ui.checkBox_search_flagged.setChecked(False)

    def checkedFlaggedSearch(self):
        isApproved = bool(self.ui.checkBox_search_approved.isChecked())
        if isApproved:
            self.ui.checkBox_search_approved.setChecked(False)

    def loadSearchImageMetaInTable(self, foundlist: List[ImageData], notFoundIdList: List[str]):
        """
        Set table content after triggering button "show"
        Parameters:
            foundlist (list): list contains found ids
            notFoundIdList (list): list contains not found ids
        """
        rowCount = len(foundlist) + len(notFoundIdList)
        self.ui.tableWidge_imageMeta.setRowCount(rowCount)
        rowCounter = 0
        for row, imageData in enumerate(foundlist):
            self.ui.tableWidge_imageMeta.setItem(row, 0, qt.QTableWidgetItem(imageData.getName()))
            self.ui.tableWidge_imageMeta.setItem(row, 1, qt.QTableWidgetItem("Yes"))
            self.ui.tableWidge_imageMeta.setItem(row, 2, qt.QTableWidgetItem(str(imageData.isSegemented())))
            rowCounter += 1

        for row, notFoundId in enumerate(notFoundIdList):
            self.ui.tableWidge_imageMeta.setItem(rowCounter, 0, qt.QTableWidgetItem(notFoundId))
            self.ui.tableWidge_imageMeta.setItem(rowCounter, 1, qt.QTableWidgetItem("No"))
            self.ui.tableWidge_imageMeta.setItem(rowCounter, 2, qt.QTableWidgetItem("No"))
            rowCounter += 1

        self.ui.btn_show_image.enabled = True

    def loadFirstImage(self):
        self.imageCounter = 0
        self.currentImageData = self.listImageData[self.imageCounter]
        self.loadNextImage(self.currentImageData)
        self.updateHorizontalSlider()

    def showSearchedImage(self):
        """
        displays dicom & segmentation to corresponding selected row in listed ids
        """
        selectedRow = self.ui.tableWidge_imageMeta.currentRow()
        if selectedRow == -1:
            logging.warn(f"{self.getCurrentTime()}: Selected row [row number = {selectedRow}]is not valid")
            return
        selectedImageId = self.ui.tableWidge_imageMeta.item(selectedRow, 0).text()

        if selectedImageId not in self.idToimageData:
            logging.info(f"{self.getCurrentTime()}: Selected image id [id = {selectedImageId}] was not found")
            return
        self.currentImageData = self.idToimageData[selectedImageId]
        self.loadNextImage(self.currentImageData)

    def removeAllWhiteSpaces(self, strChain) -> str:
        """
        removes white spaces within string
        """
        pattern = r"\s+"
        return re.sub(pattern, "", strChain)

    def getIdsFromString(self, idStr: str) -> List[str]:
        """
        parses string which contains comma seperated ids
        Parameters:
            idStr (str): string which contains comma seperated ids
        Returns:
            list: contains ids
        """
        cleanedStr = self.removeAllWhiteSpaces(idStr)
        idsList = cleanedStr.split(",")
        return list(dict.fromkeys(idsList))  # remove all duplicates

    # Section: Dicom stream
    # Button: Approve
    def approveSegmentation(self):
        statusApproved = self.ui.btn_approved.isChecked()
        statusFlagged = self.ui.btn_mark_revision.isChecked()

        if statusFlagged or self.getCurrentMetaStatus() == self.STATUS.FLAGGED:
            self.ui.btn_mark_revision.setChecked(False)
            self.ui.btn_mark_revision.setDown(False)
            self.ui.btn_mark_revision.setStyleSheet(self.colorDarkGrayButton)

        if statusApproved:
            self.setCurrentMetaStatus(status=self.STATUS.APPROVED)
            self.ui.btn_approved.setChecked(True)
            self.ui.btn_approved.setStyleSheet(self.colorLightGreenButton)
            self.ui.btn_mark_revision.setStyleSheet(self.colorDarkGrayButton)
            self.ui.lineEdit_status.setStyleSheet(self.colorLightGreenButton)
        else:
            self.setCurrentMetaStatus(status="")
            self.resetButtonsOfApproveOrFlag()
        self.updateDisplayImageMetaData()

    # Button: Flagge
    def flagSegmentation(self):
        statusApproved = self.ui.btn_approved.isChecked()
        statusFlagged = self.ui.btn_mark_revision.isChecked()

        if statusApproved or self.getCurrentMetaStatus() == self.STATUS.APPROVED:
            self.ui.btn_approved.setChecked(False)
            self.ui.btn_approved.setDown(False)
            self.ui.btn_approved.setStyleSheet(self.colorDarkGrayButton)
        if statusFlagged:
            self.setCurrentMetaStatus(status=self.STATUS.FLAGGED)
            self.ui.btn_mark_revision.setChecked(True)
            self.ui.btn_mark_revision.setStyleSheet(self.colorLightGreenButton)
            self.ui.btn_approved.setStyleSheet(self.colorDarkGrayButton)
            self.ui.lineEdit_status.setStyleSheet(self.colorLightYellow)
        else:
            self.setCurrentMetaStatus(status="")
            self.resetButtonsOfApproveOrFlag()
        self.updateDisplayImageMetaData()

    def resetButtonsOfApproveOrFlag(self):
        self.ui.btn_mark_revision.setStyleSheet("")
        self.ui.btn_approved.setStyleSheet("")
        self.ui.lineEdit_status.setStyleSheet("")

    # Button: Clear
    def clearButtons(self):
        self.ui.btn_mark_revision.setChecked(False)
        self.ui.btn_approved.setChecked(False)

        self.ui.btn_mark_revision.setDown(False)
        self.ui.btn_approved.setDown(False)

        self.resetButtonsOfApproveOrFlag()

        self.ui.btn_easy.setChecked(False)
        self.ui.btn_medium.setChecked(False)
        self.ui.btn_hard.setChecked(False)

        self.ui.btn_easy.setDown(False)
        self.ui.btn_medium.setDown(False)
        self.ui.btn_hard.setDown(False)

        self.resetButtonOfDifficulty()

    def disableButtons(self):
        self.ui.btn_easy.setDown(False)
        self.ui.btn_medium.setDown(False)
        self.ui.btn_hard.setDown(False)

    def setDifficultyButtonAccordingColorAndChecked(self, difficulty: str):
        if difficulty == self.LEVEL.EASY:
            self.ui.btn_easy.setStyleSheet(self.colorGreenEasyButton)
            self.ui.lineEdit_level.setStyleSheet(self.colorGreenEasyButton)

            self.ui.btn_medium.setChecked(False)
            self.ui.btn_hard.setChecked(False)

            self.ui.btn_medium.setDown(False)
            self.ui.btn_hard.setDown(False)

            self.ui.btn_medium.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_hard.setStyleSheet(self.colorDarkGrayButton)

        elif difficulty == self.LEVEL.MEDIUM:
            self.ui.btn_medium.setStyleSheet(self.colorYellowMediumButton)
            self.ui.lineEdit_level.setStyleSheet(self.colorYellowMediumButton)

            self.ui.btn_easy.setChecked(False)
            self.ui.btn_hard.setChecked(False)

            self.ui.btn_easy.setDown(False)
            self.ui.btn_hard.setDown(False)

            self.ui.btn_easy.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_hard.setStyleSheet(self.colorDarkGrayButton)

        elif difficulty == self.LEVEL.HARD:
            self.ui.btn_hard.setStyleSheet(self.colorRedHardButton)
            self.ui.lineEdit_level.setStyleSheet(self.colorRedHardButton)

            self.ui.btn_easy.setChecked(False)
            self.ui.btn_medium.setChecked(False)

            self.ui.btn_easy.setDown(False)
            self.ui.btn_medium.setDown(False)

            self.ui.btn_easy.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_medium.setStyleSheet(self.colorDarkGrayButton)

    # Button: Easy
    def setEasy(self):
        levelEasy = self.ui.btn_easy.isChecked()
        if levelEasy:
            self.setCurrentMetaLevel(level=self.LEVEL.EASY)
            self.setDifficultyButtonAccordingColorAndChecked(difficulty=self.LEVEL.EASY)
            self.ui.lineEdit_level.setText(self.getCurrentMetaLevel())

        if levelEasy is False and self.getCurrentMetaLevel() == self.LEVEL.EASY:
            self.setCurrentMetaLevel(level="")
            self.resetButtonOfDifficulty()
            self.ui.lineEdit_level.setStyleSheet("")

        self.updateDisplayImageMetaData()

    # Button: Medium
    def setMedium(self):
        levelMedium = self.ui.btn_medium.isChecked()
        if levelMedium:
            self.setCurrentMetaLevel(level=self.LEVEL.MEDIUM)
            self.setDifficultyButtonAccordingColorAndChecked(difficulty=self.LEVEL.MEDIUM)
            self.ui.lineEdit_level.setText(self.getCurrentMetaLevel())

        if levelMedium is False and self.getCurrentMetaLevel() == self.LEVEL.MEDIUM:
            self.setCurrentMetaLevel(level="")
            self.resetButtonOfDifficulty()
            self.ui.lineEdit_level.setStyleSheet("")

        self.updateDisplayImageMetaData()

    # Button: Hard
    def setHard(self):
        levelHard = self.ui.btn_hard.isChecked()

        if levelHard:
            self.setCurrentMetaLevel(level=self.LEVEL.HARD)
            self.setDifficultyButtonAccordingColorAndChecked(difficulty=self.LEVEL.HARD)
            self.ui.lineEdit_level.setText(self.getCurrentMetaLevel())

        if levelHard is False and self.getCurrentMetaLevel() == self.LEVEL.HARD:
            self.setCurrentMetaLevel(level="")
            self.resetButtonOfDifficulty()
            self.ui.lineEdit_level.setStyleSheet("")

        self.updateDisplayImageMetaData()

    def resetButtonOfDifficulty(self):
        self.ui.btn_easy.setStyleSheet(self.colorGreenEasyButton)
        self.ui.btn_medium.setStyleSheet(self.colorYellowMediumButton)
        self.ui.btn_hard.setStyleSheet(self.colorRedHardButton)

    # Button: Next
    def getNextSegmentation(self):
        """
        after triggering next button:
          1. persist meta data in monai server
          2. update progess bar
          3. load next dicom & segmentation
        """

        self.persistMetaInMonaiServer()

        # Re process Meta Data after image data was persisted
        self.reloadOverallStatistic()
        # Request Next Image
        self.imageCounter += 1

        if self.imageCounter >= len(self.listImageData):
            message = f"{self.getCurrentTime()}: End of list has been reached."
            slicer.util.warningDisplay(message)
            self.imageCounter = len(self.listImageData) - 1
            return
        self.updateHorizontalSlider()
        self.currentImageData = self.listImageData[self.imageCounter]

        # Displays Next Image
        self.loadNextImage(self.currentImageData)
        self.resetSegmentationEditorTools()
        self.activateSegmentatorEditor(activated=False)

    # Monai Server: Put
    def persistMetaInMonaiServer(self):
        """
        Sends the updated meta data of dicom and segmentation to monai-server
        Monai-server incorporates that information into datastore.json file
        """
        self.logic.updateLabelInfo(
            imageData=self.currentImageData,
            versionTag=self.getCurrentLabelVersion(),
            status=self.getCurrentMetaStatus(),
            level=self.getCurrentMetaLevel(),
            approvedBy=self.selectedReviewer,
            comment=self.getCurrentComment(),
        )

    # Button: Previouse
    def getPreviousSegmenation(self):
        """
        Loads the previous dicom and corresponding segmentation
        after useres tiggers Previous-Button
        """
        self.imageCounter -= 1
        if self.imageCounter < 0:
            message = f"{self.getCurrentTime()}: Lower limit of data set has been reached."
            slicer.util.warningDisplay(message)
            self.imageCounter = 0
            return
        self.updateHorizontalSlider()
        self.currentImageData = self.listImageData[self.imageCounter]
        self.currentImageData.display()

        self.fillComboBoxLabelVersions(self.currentImageData)
        approvedOrLatestVersionTag = self.currentImageData.getApprovedVersionTagElseReturnLatestVersion()
        self.loadNextImage(self.currentImageData, tag=approvedOrLatestVersionTag)
        self.resetSegmentationEditorTools()

    def cleanLineEditsContainingSegMeta(self):
        self.ui.lineEdit_image_id.setText("")
        self.ui.lineEdit_status.setText("")
        self.ui.lineEdit_segmentator.setText("")
        self.ui.lineEdit_level.setText("")
        self.ui.lineEdit_level.setStyleSheet("")
        self.ui.lineEdit_date.setText("")
        self.ui.plainText_comment.setPlainText("")

    def displayImageMetaData(self, imageData: ImageData, currentLabelVersion: str):
        """
        Displays meta info of dicom and segmentation in the info box on slicer

        Parameters:
          imageData (ImageData): Contains meta data (of dicom and segmenation)
        """
        self.cleanLineEditsContainingSegMeta()
        self.clearButtons()

        self.setCurrentMetaStatus(status=imageData.getStatus(currentLabelVersion))

        self.fillLineEditsWithSegmenationMeta(imageData, currentLabelVersion)
        self.setMetaButtonsAccordingToImageData(imageData, currentLabelVersion)

        self.setCurrentMetaLevel(level=imageData.getLevel(currentLabelVersion))

    def setMetaButtonsAccordingToImageData(self, imageData: ImageData, currentLabelVersion: str):
        finalLevel = imageData.getLevel(currentLabelVersion)
        if finalLevel != "":
            self.activateBtnLevelOfDifficulty(finalLevel)

        if imageData.isApprovedVersion(currentLabelVersion):
            self.activateBtnApproved(True)

        if imageData.isFlagged(currentLabelVersion):
            self.activateBtnApproved(False)

    def fillLineEditsWithSegmenationMeta(self, imageData: ImageData, currentLabelVersion: str):

        logging.info(f"==== currentLabelVersion: {currentLabelVersion}")
        logging.info(f"==== getName: {imageData.getName()}")
        logging.info(f"==== getClientId: {imageData.getClientId(currentLabelVersion)}")
        logging.info(f"==== getTime: {imageData.getTimeOfAnnotation()}")
        logging.info(f"==== getStatus: {imageData.getStatus(currentLabelVersion)}")
        logging.info(f"==== getComment: {imageData.getComment(currentLabelVersion)}")
        logging.info(f"==== getLevel: {imageData.getLevel(currentLabelVersion)}")
        logging.info(f"==== edtitingTme: {imageData.getTimeOfEditing(currentLabelVersion)}")

        name = imageData.getName()
        annotator = imageData.getClientId(currentLabelVersion)
        editor = imageData.getApprovedBy(currentLabelVersion)
        edtitingTme = imageData.getTimeOfEditing(currentLabelVersion)
        annotationTime = imageData.getTimeOfAnnotation()
        status = imageData.getStatus(currentLabelVersion)
        comment = imageData.getComment(currentLabelVersion)

        self.ui.lineEdit_image_id.setText(name)
        self.ui.lineEdit_segmentator.setText(annotator)
        self.ui.lineEdit_editor.setText(editor)
        self.ui.lineEdit_editing_date.setText(edtitingTme)
        self.ui.lineEdit_date.setText(annotationTime)
        self.ui.lineEdit_status.setText(status)
        self.ui.plainText_comment.setPlainText(comment)

        finalLevel = imageData.getLevel(currentLabelVersion)
        self.ui.lineEdit_level.setText(finalLevel)

    def activateBtnLevelOfDifficulty(self, finalLevel):
        if finalLevel == self.LEVEL.EASY:
            self.ui.btn_easy.setDown(True)
            self.ui.btn_easy.setChecked(True)

            self.ui.btn_easy.setStyleSheet(self.colorGreenEasyButton)
            self.ui.btn_medium.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_hard.setStyleSheet(self.colorDarkGrayButton)

            self.ui.lineEdit_level.setStyleSheet(self.colorGreenEasyButton)
            self.setEasy()

        elif finalLevel == self.LEVEL.MEDIUM:
            self.ui.btn_medium.setDown(True)
            self.ui.btn_medium.setChecked(True)

            self.ui.btn_easy.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_medium.setStyleSheet(self.colorYellowMediumButton)
            self.ui.btn_hard.setStyleSheet(self.colorDarkGrayButton)

            self.ui.lineEdit_level.setStyleSheet(self.colorYellowMediumButton)
            self.setMedium()

        elif finalLevel == self.LEVEL.HARD:
            self.ui.btn_hard.setDown(True)
            self.ui.btn_hard.setChecked(True)

            self.ui.btn_easy.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_medium.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_hard.setStyleSheet(self.colorRedHardButton)

            self.ui.lineEdit_level.setStyleSheet(self.colorRedHardButton)
            self.setHard()

    def activateBtnApproved(self, activated: bool):
        self.ui.btn_mark_revision.setChecked(not activated)
        self.ui.btn_mark_revision.setDown(not activated)
        self.ui.btn_approved.setChecked(activated)
        self.ui.btn_approved.setDown(activated)
        if activated:
            self.ui.btn_approved.setStyleSheet(self.colorLightGreenButton)
            self.ui.btn_mark_revision.setStyleSheet(self.colorDarkGrayButton)
            self.ui.lineEdit_status.setStyleSheet(self.colorLightGreenButton)
        else:
            self.ui.btn_approved.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_mark_revision.setStyleSheet(self.colorLightGreenButton)
            self.ui.lineEdit_status.setStyleSheet(self.colorLightYellow)

    def updateDisplayImageMetaData(self):
        """
        Displays updated level (easy, medium, hard)
        in the info box on slicer
        """
        self.ui.lineEdit_status.setText(self.getCurrentMetaStatus())

    def loadNextImage(self, imageData: ImageData, tag=""):
        """
        Loads original Dicom image and Segmentation into slicer window
        Parameters:
          imageData (ImageData): Contains meta data (of dicom and segmenation)
                                which is required for rest request to monai server
                                in order to get dicom and segmenation (.nrrd).
        """
        slicer.mrmlScene.Clear()
        self.clearInformationFields()

        if tag == "":
            tag = self.currentImageData.getApprovedVersionTagElseReturnLatestVersion()
        if tag == "":
            tag = self.getCurrentLabelVersion()
        logging.warn(f"{self.getCurrentTime()} Loading image (id='{imageData.getName()}', tag='{tag}')")
        self.disableDifficultyButtons(tag=tag)
        self.displayImageMetaData(imageData, tag)

        self.logic.loadDicomAndSegmentation(imageData, tag)

        if imageData.getStatus() != self.STATUS.NOT_SEGMENTED:
            self.displayLabelOfSegmentation()

        if self.currentImageId is not imageData.getName():
            self.currentImageId = imageData.getName()
            self.fillComboBoxLabelVersions(imageData)

    def fillComboBoxLabelVersions(self, imageData: ImageData):
        self.isSelectableByLabelVersion = False
        self.ui.comboBox_label_version.clear()
        labelVersions = imageData.getVersionNames()
        approvedVersion = ""
        labelVersion = ""
        for labelVersion in labelVersions:
            if imageData.isApprovedVersion(versionTag=labelVersion) is True:
                labelVersion = "{} ({})".format(labelVersion, "approved")
                approvedVersion = labelVersion
            self.ui.comboBox_label_version.addItem(labelVersion)
        if approvedVersion != "":
            self.setVersionTagInComboBox(approvedVersion)
        elif labelVersion != "":
            self.setVersionTagInComboBox(labelVersion)
        self.isSelectableByLabelVersion = True

    def setVersionTagInComboBox(self, versionTag="", currentImageData=None):
        if self.isBlank(versionTag):
            return
        if (currentImageData is not None) and (currentImageData.isApprovedVersion(versionTag=versionTag) is True):
            versionTag = "{} ({})".format(versionTag, "approved")
        self.ui.comboBox_label_version.setCurrentText(versionTag)

    def parseSelectedVersionFromComboBox(self, versionTagString: str) -> str:
        if self.isBlank(versionTagString):
            return ""
        array = versionTagString.split()
        if len(array) == 0:
            return ""
        return array[0]

    # Sub Section: Display version selection option
    def activateSegmentatorEditor(self, activated=False):
        self.segmentEditorWidget.setMasterVolumeNodeSelectorVisible(activated)
        self.segmentEditorWidget.setSegmentationNodeSelectorVisible(activated)
        self.segmentEditorWidget.setSwitchToSegmentationsButtonVisible(activated)
        self.segmentEditorWidget.unorderedEffectsVisible = activated
        self.segmentEditorWidget.setReadOnly(not activated)

    def displayEditorTools(self):
        isCheckedForEdit = self.ui.btn_edit_label.isChecked()
        if isCheckedForEdit:
            self.clearButtons()
            self.ui.btn_edit_label.setText("Reset current label edit")
            self.cleanLineEditsContainingSegMetaWhenStartEditing()
            self.activatedEditorTools()
        else:
            self.deactivatedEditorTools()
            self.loadNextImage(imageData=self.currentImageData, tag=self.getCurrentLabelVersion())

    def cleanLineEditsContainingSegMetaWhenStartEditing(self):
        self.setCurrentMetaStatus(status="")
        self.setCurrentComment(comment="")
        self.ui.lineEdit_status.setText("")
        self.ui.lineEdit_editing_date.setText("")
        self.ui.plainText_comment.setPlainText("")

    def clearInformationFields(self):
        self.setCurrentMetaStatus(status="")
        self.setCurrentComment(comment="")
        self.setCurrentMetaLevel(level="")

    def activatedEditorTools(self):
        self.activateSegmentatorEditor(activated=True)
        self.ui.btn_save_new_version.setChecked(False)
        self.ui.btn_overwrite_version.setChecked(False)
        self.ui.btn_delete_version.setChecked(False)

        self.ui.btn_save_new_version.show()
        self.ui.btn_save_new_version.setStyleSheet("")

        self.ui.btn_overwrite_version.show()
        self.ui.btn_overwrite_version.setStyleSheet("")

        self.ui.btn_delete_version.show()
        self.ui.btn_delete_version.setStyleSheet("")
        self.ui.btn_update_version.show()

        self.ui.btn_next.enabled = False
        self.ui.btn_previous.enabled = False
        self.ui.btn_easy.enabled = False
        self.ui.btn_medium.enabled = False
        self.ui.btn_hard.enabled = False
        self.ui.btn_mark_revision.enabled = False
        self.ui.btn_approved.enabled = False

    def deactivatedEditorTools(self):
        self.activateSegmentatorEditor(activated=False)
        self.ui.btn_save_new_version.setChecked(False)
        self.ui.btn_overwrite_version.setChecked(False)
        self.ui.btn_delete_version.setChecked(False)

        self.ui.btn_save_new_version.hide()
        self.ui.btn_overwrite_version.hide()
        self.ui.btn_delete_version.hide()
        self.ui.btn_update_version.hide()
        self.ui.btn_update_version.enabled = False

        self.ui.btn_next.enabled = True
        self.ui.btn_previous.enabled = True
        self.ui.btn_easy.enabled = True
        self.ui.btn_medium.enabled = True
        self.ui.btn_hard.enabled = True
        self.ui.btn_mark_revision.enabled = True
        self.ui.btn_approved.enabled = True

        self.ui.btn_update_version.setText("Confirm")
        self.ui.btn_edit_label.setText("Start label edit")

    def resetSegmentationEditorTools(self):
        self.ui.btn_edit_label.setChecked(False)
        self.deactivatedEditorTools()

    def hideEditingSelectionOption(self, isHidden: bool):
        if isHidden:
            self.ui.btn_edit_label.hide()
        else:
            self.ui.btn_edit_label.show()

    # Section: Display label
    def addSegmentator(self):
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.segmentEditorWidget.setEffectNameOrder([])

    def displayLabelOfSegmentation(self):
        self.selectParameterNode()
        self.getDefaultMasterVolumeNodeID()
        self.segmentEditorWidget.SegmentationNodeComboBox.setCurrentNodeIndex(0)
        self.segmentEditorWidget.MasterVolumeNodeComboBox.setCurrentNodeIndex(0)

    def selectParameterNode(self):
        # Select parameter set node if one is found in the scene, and create one otherwise
        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if segmentEditorNode is None:
            segmentEditorNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorNode.UnRegister(None)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        self.segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

    def getDefaultMasterVolumeNodeID(self):
        layoutManager = slicer.app.layoutManager()
        firstForegroundVolumeID = None
        # Use first background volume node in any of the displayed layouts.
        # If no beackground volume node is in any slice view then use the first
        # foreground volume node.
        for sliceViewName in layoutManager.sliceViewNames():
            sliceWidget = layoutManager.sliceWidget(sliceViewName)
            if not sliceWidget:
                continue
            compositeNode = sliceWidget.mrmlSliceCompositeNode()
            if compositeNode.GetBackgroundVolumeID():
                return compositeNode.GetBackgroundVolumeID()
            if compositeNode.GetForegroundVolumeID() and not firstForegroundVolumeID:
                firstForegroundVolumeID = compositeNode.GetForegroundVolumeID()
        # No background volume was found, so use the foreground volume (if any was found)
        return firstForegroundVolumeID

    def getVersionName(self) -> str:
        if self.setOverwriteCurrentVersion():
            return self.getCurrentLabelVersion()
        else:
            return self.currentImageData.getNewVersionName()

    def persistEditedSegmentation(self, newVersionName: str):
        self.currentImageData.updateSegmentationMetaByVerionTag(
            tag=newVersionName,
            status=self.getCurrentMetaStatus(),
            level=self.getCurrentMetaLevel(),
            approvedBy=self.selectedReviewer,
            comment=self.getCurrentComment(),
        )

        segmentationNode = self.segmentEditorWidget.segmentationNode()
        self.tmpdir = slicer.util.tempDirectory("slicer-monai-reviewer")
        label_in = tempfile.NamedTemporaryFile(suffix=".nrrd", dir=self.tmpdir).name
        slicer.util.saveNode(segmentationNode, label_in)

        self.logic.saveLabelInMonaiServer(imageData=self.currentImageData, label_in=label_in, tag=newVersionName)

    def deleteLabelByVersionTag(self):
        imageVersionTag = self.getCurrentLabelVersion()
        if self.isBlank(imageVersionTag):
            return
        if self.currentImageData.hasVersionTag(versionTag=imageVersionTag) is False:
            return
        self.logic.deleteLabelByVersionTag(imageData=self.currentImageData, versionTag=imageVersionTag)

    def updateAfterEditingSegmentation(self):
        imageVersionTag = self.getCurrentLabelVersion()
        setToSave = bool(self.ui.btn_save_new_version.isChecked())
        setToOverwrite = bool(self.ui.btn_overwrite_version.isChecked())
        setToDelete = bool(self.ui.btn_delete_version.isChecked())
        newLabelNameCreated = ""
        if setToSave:
            newLabelNameCreated = self.currentImageData.getNewVersionName()
            self.persistEditedSegmentation(newVersionName=newLabelNameCreated)

        elif setToOverwrite:
            if (imageVersionTag == self.LABEL.FINAL) or (imageVersionTag == self.LABEL.ORIGINAL):
                warningMessage: str = "Initial Segmentation with label 'final' or 'original' \ncannot be overwritten.\n Please save current edit as new version."
                slicer.util.warningDisplay(warningMessage)
                logging.warn(warningMessage)
                return

            self.persistEditedSegmentation(newVersionName=imageVersionTag)

        elif setToDelete:

            if (imageVersionTag == self.LABEL.FINAL) or (imageVersionTag == self.LABEL.ORIGINAL):
                warningMessage: str = "Initial Segmentation with label 'final' or 'original' \ncannot be deleted."
                slicer.util.warningDisplay(warningMessage)
                logging.warn(warningMessage)
                return

            self.deleteLabelByVersionTag()

        self.resetSegmentationEditorTools()
        self.reloadImageAfterEditingLabel()
        if newLabelNameCreated != "":
            self.setVersionTagInComboBox(versionTag=newLabelNameCreated)

    def reloadImageAfterEditingLabel(self):
        imageId = self.currentImageData.getFileName()
        latestVersion = self.currentImageData.getLatestVersionTag()
        logging.info(f"{self.getCurrentTime()}: Loading image (id='{imageId}') with version tag = '{latestVersion}'")
        self.loadNextImage(imageData=self.currentImageData, tag=latestVersion)
        self.fillComboBoxLabelVersions(self.currentImageData)

    def processDataStoreRecords(self):
        serverUrl: str = self.ui.comboBox_server_url.currentText
        result: bool = self.logic.initMetaDataProcessing()
        if result is False:
            warningMessage = (
                "Request for datastore-info failed.\nPlease check if server address is correct \n('{}')!".format(
                    serverUrl
                )
            )
            slicer.util.warningDisplay(warningMessage)
            logging.warn(warningMessage)
            return
        logging.info(f"{self.getCurrentTime()}: Successfully processed all records in datastore.")

    def reloadOverallStatistic(self):
        if self.reviewersModeIsActive:
            self.processDataStoreRecords()
            self.setProgessBar()
            self.setProgressBarOfAll()

    def displayAdditionalMetaIfEdited(self, tag: str):
        if tag == self.LABEL.FINAL or tag == self.LABEL.ORIGINAL:
            self.ui.lineEdit_editor.hide()
            self.ui.lineEdit_editing_date.hide()
            self.ui.label_editor.hide()
            self.ui.label_editing_date.hide()
        else:
            self.ui.lineEdit_editor.show()
            self.ui.lineEdit_editing_date.show()
            self.ui.label_editor.show()
            self.ui.label_editing_date.show()

    def isBlank(self, string) -> bool:
        return not (string and string.strip())


class MONAILabelReviewerLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.temp_dir = None
        self.imageDataController: ImageDataController = ImageDataController()

    # Section: Server
    def getServerUrl(self) -> str:
        return self.imageDataController.getServerUrl()

    def getCurrentTime(self) -> datetime:
        return datetime.datetime.now()

    def connectToMonaiServer(self, serverUrl: str) -> bool:
        return self.imageDataController.connectToMonaiServer(serverUrl)

    def getMapIdToImageData(self) -> Dict[str, ImageData]:
        """
        Returns dictionary (Dict[str:ImageData]) which maps id to Imagedata-object
        """
        return self.imageDataController.getMapIdToImageData()

    def initMetaDataProcessing(self) -> bool:
        return self.imageDataController.initMetaDataProcessing()

    def getStatistics(self) -> ImageDataStatistics:
        return self.imageDataController.getStatistics()

    def getClientIds(self) -> List[str]:
        return self.imageDataController.getClientIds()

    def getReviewers(self) -> List[str]:
        return self.imageDataController.getReviewers()

    # Section: Loading images
    def getAllImageData(
        self, segmented: str, isNotSegmented: str, isApproved: bool, isFlagged: bool
    ) -> List[ImageData]:
        return self.imageDataController.getAllImageData(segmented, isNotSegmented, isApproved, isFlagged)

    def getImageDataByClientId(self, selectedClientId: str, isApproved: bool, isFlagged: bool) -> List[ImageData]:
        return self.imageDataController.getImageDataByClientId(selectedClientId, isApproved, isFlagged)

    def getPercentageApproved(self, selectedClientId: str):
        percentageApprovedOfClient, idxApprovedOfClient = self.imageDataController.getPercentageApproved(
            selectedClientId
        )
        return percentageApprovedOfClient, idxApprovedOfClient

    def getPercentageSemgmentedByClient(self, selectedClientId: str):
        percentageSemgmentedByClient, idxSegmentedByClient = self.imageDataController.getPercentageSemgmentedByClient(
            selectedClientId
        )
        return percentageSemgmentedByClient, idxSegmentedByClient

    # Section: Search Image
    def getMultImageDataByIds(self, idList: List[str]) -> Dict[str, ImageData]:
        return self.imageDataController.getMultImageDataByIds(idList)

    def searchByAnnotatorReviewer(
        self, selectedAnnotator: str, selectedReviewer: str, isApproved: bool, isFlagged: bool
    ) -> Dict[str, ImageData]:
        return self.imageDataController.searchByAnnotatorReviewer(
            selectedAnnotator, selectedReviewer, isApproved, isFlagged
        )

    def searchByLevel(self, isEasy: bool, isMedium: bool, isHard: bool) -> Dict[str, ImageData]:
        return self.imageDataController.getImageDataByLevel(isEasy=isEasy, isMedium=isMedium, isHard=isHard)

    def updateImageData(
        self, imageData: ImageData, versionTag: str, status: str, level: str, approvedBy: str, comment: str
    ) -> dict:
        """
        update meta data in information box
        Returns: jsonDict: json dictionary which contains updated meta data
        """
        imageId = imageData.getName()
        isEqual = imageData.isEqualSegmentationMeta(
            tag=versionTag, status=status, level=level, approvedBy=approvedBy, comment=comment
        )
        if isEqual:
            logging.info(f"{self.getCurrentTime()}: No changes for image (id='{imageId}')")
            return ""

        imageData.updateSegmentationMetaByVerionTag(
            tag=versionTag, status=status, level=level, approvedBy=approvedBy, comment=comment
        )
        jsonDict = imageData.getMetaByVersionTag(tag=versionTag)

        if jsonDict is None:
            logging.info(f"{self.getCurrentTime()}: No update for Image (id='{imageId}')")
            return ""
        logging.info(f"{self.getCurrentTime()}: Successfully updated Image (id='{imageId}')")
        return jsonDict

    # Section: Dicom stream
    def updateLabelInfo(
        self, imageData: ImageData, versionTag: str, status: str, level: str, approvedBy: str, comment: str
    ):
        imageId = imageData.getName()
        updatedMetaJson = self.updateImageData(imageData, versionTag, status, level, approvedBy, comment)
        if updatedMetaJson == "":
            logging.info(
                "{} :  Image update (id='{}', version tag='{}') is empty".format(
                    self.getCurrentTime(), imageId, versionTag
                )
            )
            return

        logging.info(f"{self.getCurrentTime()} : Image update (id='{imageId}', version tag='{versionTag}')")
        logging.info(updatedMetaJson)
        self.imageDataController.updateLabelInfoOfAllVersionTags(
            imageData=imageData, versionTag=versionTag, level=level, updatedMetaJson=updatedMetaJson
        )

    def loadDicomAndSegmentation(self, imageData: ImageData, tag: str):
        """
        Loads original Dicom image and Segmentation into slicer window
        Parameters:
          imageData (ImageData): Contains meta data (of dicom and segmenation)
                                 which is required for rest request to monai server
                                 in order to get dicom and segmenation (.nrrd).
        """
        # Request dicom
        image_name = imageData.getFileName()
        image_id = imageData.getName()
        node_name = imageData.getNodeName()
        checksum = imageData.getCheckSum()
        logging.info(
            "{}: Request Data  image_name='{}', node_name='{}', image_id='{}', checksum='{}'".format(
                self.getCurrentTime(), image_name, node_name, image_id, checksum
            )
        )

        self.requestDicomImage(image_id, image_name, node_name, checksum)
        self.setTempFolderDir()

        # Request segmentation
        if imageData.isSegemented():
            segmentationFileName = imageData.getSegmentationFileName()
            img_blob = self.imageDataController.reuqestSegmentation(image_id, tag)
            destination = self.storeSegmentation(img_blob, segmentationFileName, self.temp_dir.name)
            self.displaySegmention(destination)
            os.remove(destination)
            logging.info(f"{self.getCurrentTime()}: Removed file at {destination}")

    def storeSegmentation(
        self, response: requests.models.Response, segmentationFileName: str, tempDirectory: str
    ) -> str:
        """
        stores loaded segmenation temporarily in local directory
        Parameters:
            response (requests.models.Response): contains segmentation data
            image_id (str): image id of segmentation
        """
        segmentation = response.content
        destination = self.getPathToStore(segmentationFileName, tempDirectory)
        with open(destination, "wb") as img_file:
            img_file.write(segmentation)
        logging.info(f"{self.getCurrentTime()}: Image segmentation is stored temoparily in: {destination}")
        return destination

    def getPathToStore(self, segmentationFileName: str, tempDirectory: str) -> str:
        return tempDirectory + "/" + segmentationFileName

    def displaySegmention(self, destination: str):
        """
        Displays the segmentation in slicer window
        """
        segmentation = slicer.util.loadSegmentation(destination)

    def requestDicomImage(self, image_id: str, image_name: str, node_name: str, checksum: str):
        download_uri = self.imageDataController.getDicomDownloadUri(image_id)
        sampleDataLogic = SampleData.SampleDataLogic()
        _volumeNode = sampleDataLogic.downloadFromURL(
            nodeNames=node_name, fileNames=image_name, uris=download_uri, checksums=checksum
        )[0]

    def setTempFolderDir(self):
        """
        Create temporary dirctory to store the downloaded segmentation (.nrrd)
        """
        if self.temp_dir is None:
            self.temp_dir = tempfile.TemporaryDirectory()
        logging.info(f"{self.getCurrentTime()}: Temporary Directory: '{self.temp_dir.name}'")

    def saveLabelInMonaiServer(self, imageData: ImageData, label_in: str, tag: str):
        imageName = imageData.getName()
        params = imageData.obtainUpdatedParams(tag)
        self.imageDataController.saveLabelInMonaiServer(imageName, label_in, tag, params)

    def deleteLabelByVersionTag(self, imageData: ImageData, versionTag: str) -> bool:
        imageId = imageData.getName()
        imageData.deleteVersionName(versionTag)
        successfullyDeleted: bool = self.imageDataController.deleteLabelByVersionTag(imageId, versionTag)
        return successfullyDeleted


class MONAILabelReviewerTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_MONAILabelReviewer1()

    def test_MONAILabelReviewer1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")
