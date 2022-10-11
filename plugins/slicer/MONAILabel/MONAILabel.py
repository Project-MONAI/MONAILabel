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

import json
import logging
import os
import shutil
import tempfile
import time
import traceback
from collections import OrderedDict
from urllib.parse import quote_plus

import ctk
import qt
import SampleData
import SimpleITK as sitk
import sitkUtils
import slicer
import vtk
import vtkSegmentationCore
from MONAILabelLib import GenericAnatomyColors, MONAILabelClient
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


class MONAILabel(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MONAILabel"
        self.parent.categories = ["Active Learning"]
        self.parent.dependencies = []
        self.parent.contributors = ["NVIDIA, KCL"]
        self.parent.helpText = """
Active Learning solution.
See more information in <a href="https://github.com/Project-MONAI/MONAILabel">module documentation</a>.
"""
        self.parent.acknowledgementText = """
Developed by NVIDIA, KCL
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", self.initializeAfterStartup)

    def initializeAfterStartup(self):
        if not slicer.app.commandOptions().noMainWindow:
            self.settingsPanel = MONAILabelSettingsPanel()
            slicer.app.settingsDialog().addPanel("MONAI Label", self.settingsPanel)


class _ui_MONAILabelSettingsPanel:
    def __init__(self, parent):
        vBoxLayout = qt.QVBoxLayout(parent)

        # settings
        groupBox = ctk.ctkCollapsibleGroupBox()
        groupBox.title = "MONAI Label Server"
        groupLayout = qt.QFormLayout(groupBox)

        serverUrl = qt.QLineEdit()
        groupLayout.addRow("Server address:", serverUrl)
        parent.registerProperty("MONAILabel/serverUrl", serverUrl, "text", str(qt.SIGNAL("textChanged(QString)")))

        serverUrlHistory = qt.QLineEdit()
        groupLayout.addRow("Server address history:", serverUrlHistory)
        parent.registerProperty(
            "MONAILabel/serverUrlHistory", serverUrlHistory, "text", str(qt.SIGNAL("textChanged(QString)"))
        )

        fileExtension = qt.QLineEdit()
        fileExtension.setText(".nii.gz")
        fileExtension.toolTip = "Default extension for uploading images/labels"
        groupLayout.addRow("File Extension:", fileExtension)
        parent.registerProperty(
            "MONAILabel/fileExtension", fileExtension, "text", str(qt.SIGNAL("textChanged(QString)"))
        )

        clientId = qt.QLineEdit()
        clientId.setText("user-xyz")
        clientId.toolTip = "Client/User ID that will be sent to MONAI Label server for reference"
        groupLayout.addRow("Client/User-ID:", clientId)
        parent.registerProperty("MONAILabel/clientId", clientId, "text", str(qt.SIGNAL("textChanged(QString)")))

        autoRunSegmentationCheckBox = qt.QCheckBox()
        autoRunSegmentationCheckBox.checked = False
        autoRunSegmentationCheckBox.toolTip = (
            "Enable this option to auto run segmentation if pre-trained model exists when Next Sample is fetched"
        )
        groupLayout.addRow("Auto-Run Pre-Trained Model:", autoRunSegmentationCheckBox)
        parent.registerProperty(
            "MONAILabel/autoRunSegmentationOnNextSample",
            ctk.ctkBooleanMapper(autoRunSegmentationCheckBox, "checked", str(qt.SIGNAL("toggled(bool)"))),
            "valueAsInt",
            str(qt.SIGNAL("valueAsIntChanged(int)")),
        )

        autoFetchNextSampleCheckBox = qt.QCheckBox()
        autoFetchNextSampleCheckBox.checked = False
        autoFetchNextSampleCheckBox.toolTip = "Enable this option to fetch Next Sample after saving the label"
        groupLayout.addRow("Auto-Fetch Next Sample:", autoFetchNextSampleCheckBox)
        parent.registerProperty(
            "MONAILabel/autoFetchNextSample",
            ctk.ctkBooleanMapper(autoFetchNextSampleCheckBox, "checked", str(qt.SIGNAL("toggled(bool)"))),
            "valueAsInt",
            str(qt.SIGNAL("valueAsIntChanged(int)")),
        )

        autoUpdateModelCheckBox = qt.QCheckBox()
        autoUpdateModelCheckBox.checked = False
        autoUpdateModelCheckBox.toolTip = "Enable this option to auto update model after submitting the label"
        groupLayout.addRow("Auto-Update Model:", autoUpdateModelCheckBox)
        parent.registerProperty(
            "MONAILabel/autoUpdateModelV2",
            ctk.ctkBooleanMapper(autoUpdateModelCheckBox, "checked", str(qt.SIGNAL("toggled(bool)"))),
            "valueAsInt",
            str(qt.SIGNAL("valueAsIntChanged(int)")),
        )

        askForUserNameCheckBox = qt.QCheckBox()
        askForUserNameCheckBox.checked = False
        askForUserNameCheckBox.toolTip = (
            "Enable this option to ask for the user name every time the MONAILabel "
            + "extension is loaded for the first time"
        )
        groupLayout.addRow("Ask For User Name:", askForUserNameCheckBox)
        parent.registerProperty(
            "MONAILabel/askForUserName",
            ctk.ctkBooleanMapper(askForUserNameCheckBox, "checked", str(qt.SIGNAL("toggled(bool)"))),
            "valueAsInt",
            str(qt.SIGNAL("valueAsIntChanged(int)")),
        )

        allowOverlapCheckBox = qt.QCheckBox()
        allowOverlapCheckBox.checked = False
        allowOverlapCheckBox.toolTip = "Enable this option to allow overlapping segmentations"
        groupLayout.addRow("Allow Overlapping Segmentations:", allowOverlapCheckBox)
        parent.registerProperty(
            "MONAILabel/allowOverlappingSegments",
            ctk.ctkBooleanMapper(allowOverlapCheckBox, "checked", str(qt.SIGNAL("toggled(bool)"))),
            "valueAsInt",
            str(qt.SIGNAL("valueAsIntChanged(int)")),
        )
        allowOverlapCheckBox.connect("toggled(bool)", self.onUpdateAllowOverlap)

        originalLabelCheckBox = qt.QCheckBox()
        originalLabelCheckBox.checked = True
        originalLabelCheckBox.toolTip = "Enable this option to first read original label (predictions)"
        groupLayout.addRow("Original Labels:", originalLabelCheckBox)
        parent.registerProperty(
            "MONAILabel/originalLabel",
            ctk.ctkBooleanMapper(originalLabelCheckBox, "checked", str(qt.SIGNAL("toggled(bool)"))),
            "valueAsInt",
            str(qt.SIGNAL("valueAsIntChanged(int)")),
        )

        developerModeCheckBox = qt.QCheckBox()
        developerModeCheckBox.checked = False
        developerModeCheckBox.toolTip = "Enable this option to find options tab etc..."
        groupLayout.addRow("Developer Mode:", developerModeCheckBox)
        parent.registerProperty(
            "MONAILabel/developerMode",
            ctk.ctkBooleanMapper(developerModeCheckBox, "checked", str(qt.SIGNAL("toggled(bool)"))),
            "valueAsInt",
            str(qt.SIGNAL("valueAsIntChanged(int)")),
        )

        showSegmentsIn3DCheckBox = qt.QCheckBox()
        showSegmentsIn3DCheckBox.checked = False
        showSegmentsIn3DCheckBox.toolTip = "Enable this option to show segments in 3D (slow) after mask update..."
        groupLayout.addRow("Show Segments In 3D:", showSegmentsIn3DCheckBox)
        parent.registerProperty(
            "MONAILabel/showSegmentsIn3D",
            ctk.ctkBooleanMapper(showSegmentsIn3DCheckBox, "checked", str(qt.SIGNAL("toggled(bool)"))),
            "valueAsInt",
            str(qt.SIGNAL("valueAsIntChanged(int)")),
        )

        vBoxLayout.addWidget(groupBox)
        vBoxLayout.addStretch(1)

    def onUpdateAllowOverlap(self):
        if slicer.util.settingsValue("MONAILabel/allowOverlappingSegments", True, converter=slicer.util.toBool):
            if slicer.util.settingsValue("MONAILabel/fileExtension", None) != ".seg.nrrd":
                slicer.util.warningDisplay(
                    "Overlapping segmentations are only availabel with the '.seg.nrrd' file extension! "
                    + "Consider changing MONAILabel file extension."
                )


class MONAILabelSettingsPanel(ctk.ctkSettingsPanel):
    def __init__(self, *args, **kwargs):
        ctk.ctkSettingsPanel.__init__(self, *args, **kwargs)
        self.ui = _ui_MONAILabelSettingsPanel(self)


class MONAILabelWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        self.logic = None
        self._parameterNode = None
        self._volumeNode = None
        self._segmentNode = None
        self._scribblesROINode = None
        self._volumeNodes = []
        self._updatingGUIFromParameterNode = False

        self.info = {}
        self.models = OrderedDict()
        self.trainers = OrderedDict()
        self.config = OrderedDict()
        self.current_sample = None
        self.samples = {}
        self.state = {
            "SegmentationModel": "",
            "DeepgrowModel": "",
            "ScribblesMethod": "",
            "CurrentStrategy": "",
            "CurrentTrainer": "",
        }
        self.file_ext = ".nii.gz"

        self.dgPositivePointListNode = None
        self.dgPositivePointListNodeObservers = []
        self.dgNegativePointListNode = None
        self.dgNegativePointListNodeObservers = []
        self.ignorePointListNodeAddEvent = False

        self.progressBar = None
        self.tmpdir = None
        self.timer = None

        self.scribblesMode = None
        self.ignoreScribblesLabelChangeEvent = False
        self.deepedit_multi_label = False

        self.optionsSectionIndex = 0
        self.optionsNameIndex = 0

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MONAILabel.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self.onSceneEndImport)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.tmpdir = slicer.util.tempDirectory("slicer-monai-label")
        self.logic = MONAILabelLogic(self.tmpdir)

        # Set icons and tune widget properties
        self.ui.serverComboBox.lineEdit().setPlaceholderText("enter server address or leave empty to use default")
        self.ui.fetchServerInfoButton.setIcon(self.icon("refresh-icon.png"))
        self.ui.segmentationButton.setIcon(self.icon("segment.png"))
        self.ui.nextSampleButton.setIcon(self.icon("segment.png"))
        self.ui.saveLabelButton.setIcon(self.icon("save.png"))
        self.ui.trainingButton.setIcon(self.icon("training.png"))
        self.ui.stopTrainingButton.setIcon(self.icon("stop.png"))
        self.ui.uploadImageButton.setIcon(self.icon("upload.svg"))
        self.ui.importLabelButton.setIcon(self.icon("download.png"))

        self.ui.dgPositiveControlPointPlacementWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.dgPositiveControlPointPlacementWidget.placeButton().toolTip = "Select +ve points"
        self.ui.dgPositiveControlPointPlacementWidget.buttonsVisible = False
        self.ui.dgPositiveControlPointPlacementWidget.placeButton().show()
        self.ui.dgPositiveControlPointPlacementWidget.deleteButton().show()

        self.ui.dgNegativeControlPointPlacementWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.dgNegativeControlPointPlacementWidget.placeButton().toolTip = "Select -ve points"
        self.ui.dgNegativeControlPointPlacementWidget.buttonsVisible = False
        self.ui.dgNegativeControlPointPlacementWidget.placeButton().show()
        self.ui.dgNegativeControlPointPlacementWidget.deleteButton().show()

        self.ui.dgUpdateButton.setIcon(self.icon("segment.png"))

        # Connections
        self.ui.fetchServerInfoButton.connect("clicked(bool)", self.onClickFetchInfo)
        self.ui.serverComboBox.connect("currentIndexChanged(int)", self.onClickFetchInfo)
        self.ui.segmentationModelSelector.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.segmentationButton.connect("clicked(bool)", self.onClickSegmentation)
        self.ui.deepgrowModelSelector.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.nextSampleButton.connect("clicked(bool)", self.onNextSampleButton)
        self.ui.trainingButton.connect("clicked(bool)", self.onTraining)
        self.ui.stopTrainingButton.connect("clicked(bool)", self.onStopTraining)
        self.ui.saveLabelButton.connect("clicked(bool)", self.onSaveLabel)
        self.ui.uploadImageButton.connect("clicked(bool)", self.onUploadImage)
        self.ui.importLabelButton.connect("clicked(bool)", self.onImportLabel)
        self.ui.labelComboBox.connect("currentIndexChanged(int)", self.onSelectLabel)
        self.ui.scribLabelComboBox.connect("currentIndexChanged(int)", self.onSelectScribLabel)
        self.ui.dgUpdateButton.connect("clicked(bool)", self.onUpdateDeepgrow)
        self.ui.dgUpdateCheckBox.setStyleSheet("padding-left: 10px;")
        self.ui.optionsSection.connect("currentIndexChanged(int)", self.onSelectOptionsSection)
        self.ui.optionsName.connect("currentIndexChanged(int)", self.onSelectOptionsName)

        # Scribbles
        # brush and eraser icon from: https://tablericons.com/
        self.ui.scribblesMethodSelector.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.paintScribblesButton.setIcon(self.icon("paint.png"))
        self.ui.paintScribblesButton.setToolTip("Paint scribbles for selected scribble layer")
        self.ui.eraseScribblesButton.setIcon(self.icon("eraser.png"))
        self.ui.eraseScribblesButton.setToolTip("Erase scribbles for selected scribble layer")
        self.ui.updateScribblesButton.setIcon(self.icon("segment.png"))
        self.ui.updateScribblesButton.setToolTip(
            "Update label by sending scribbles to server to apply selected post processing method"
        )

        self.ui.brushSizeSlider.connect("valueChanged(double)", self.updateBrushSize)
        self.ui.brushSizeSlider.setToolTip("Change brush size for scribbles tool")
        self.ui.brush3dCheckbox.stateChanged.connect(self.on3dBrushCheckbox)
        self.ui.brush3dCheckbox.setToolTip("Use 3D brush to paint/erase in multiple slices in 3D")
        self.ui.updateScribblesButton.clicked.connect(self.onUpdateScribbles)
        self.ui.paintScribblesButton.clicked.connect(self.onPaintScribbles)
        self.ui.eraseScribblesButton.clicked.connect(self.onEraseScribbles)
        self.ui.scribblesSelector.connect("currentIndexChanged(int)", self.onSelectScribblesLabel)

        # creating editable combo box
        self.ui.scribblesSelector.addItem(self.icon("fg_green.png"), "Foreground")
        self.ui.scribblesSelector.addItem(self.icon("bg_red.png"), "Background")
        self.ui.scribblesSelector.setCurrentIndex(0)

        # ROI placement for scribbles
        self.ui.scribblesPlaceWidget.setButtonsVisible(False)
        self.ui.scribblesPlaceWidget.placeButton().show()
        self.ui.scribblesPlaceWidget.setMRMLScene(slicer.mrmlScene)

        # start with scribbles section disabled
        self.ui.scribblesCollapsibleButton.setEnabled(False)
        self.ui.scribblesCollapsibleButton.collapsed = True

        # embedded segment editor
        self.ui.embeddedSegmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.embeddedSegmentEditorWidget.setSegmentationNodeSelectorVisible(False)
        self.ui.embeddedSegmentEditorWidget.setMasterVolumeNodeSelectorVisible(False)
        self.ui.embeddedSegmentEditorWidget.setMRMLSegmentEditorNode(self.logic.get_segment_editor_node())

        # options section
        self.ui.optionsSection.addItem("infer")
        self.ui.optionsSection.addItem("train")
        self.ui.optionsSection.addItem("activelearning")
        self.ui.optionsSection.addItem("scoring")

        self.initializeParameterNode()
        self.updateServerUrlGUIFromSettings()
        # self.onClickFetchInfo()

        if slicer.util.settingsValue("MONAILabel/askForUserName", False, converter=slicer.util.toBool):
            text = qt.QInputDialog().getText(
                self.parent,
                "User Name",
                "Please enter your name:",
                qt.QLineEdit.Normal,
                slicer.util.settingsValue("MONAILabel/clientId", None),
            )
            if text:
                settings = qt.QSettings()
                settings.setValue("MONAILabel/clientId", text)

    def cleanup(self):
        self.removeObservers()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def enter(self):
        self.initializeParameterNode()
        if self._segmentNode:
            self.updateGUIFromParameterNode()

    def exit(self):
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        self.state = {
            "SegmentationModel": self.ui.segmentationModelSelector.currentText,
            "DeepgrowModel": self.ui.deepgrowModelSelector.currentText,
            "ScribblesMethod": self.ui.scribblesMethodSelector.currentText,
            "CurrentStrategy": self.ui.strategyBox.currentText,
            "CurrentTrainer": self.ui.trainerBox.currentText,
        }

        self._volumeNode = None
        self._segmentNode = None
        self._volumeNodes.clear()
        self.setParameterNode(None)
        self.current_sample = None
        self.samples.clear()
        self._scribblesROINode = None

        self.resetPointList(
            self.ui.dgPositiveControlPointPlacementWidget,
            self.dgPositivePointListNode,
            self.dgPositivePointListNodeObservers,
        )
        self.dgPositivePointListNode = None
        self.resetPointList(
            self.ui.dgNegativeControlPointPlacementWidget,
            self.dgNegativePointListNode,
            self.dgNegativePointListNodeObservers,
        )
        self.dgNegativePointListNode = None
        self.onResetScribbles()

    def resetPointList(self, markupsPlaceWidget, pointListNode, pointListNodeObservers):
        if markupsPlaceWidget.placeModeEnabled:
            markupsPlaceWidget.setPlaceModeEnabled(False)

        if pointListNode:
            slicer.mrmlScene.RemoveNode(pointListNode)
            self.removePointListNodeObservers(pointListNode, pointListNodeObservers)

    def onSceneEndClose(self, caller, event):
        if self.parent.isEntered:
            self.initializeParameterNode()

    def onSceneEndImport(self, caller, event):
        if not self._volumeNode:
            self.updateGUIFromParameterNode()

    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def monitorTraining(self):
        status = self.isTrainingRunning(check_only=False)
        if status and status.get("status") == "RUNNING":
            info = self.logic.info()
            train_stats = info.get("train_stats")
            if not train_stats:
                return

            train_stats = next(iter(train_stats.values()))

            current = 0 if train_stats.get("total_time") else train_stats.get("epoch", 1)
            total = train_stats.get("total_epochs", 1)
            percent = max(1, 100 * current / total)
            if self.ui.trainingProgressBar.value != percent:
                self.ui.trainingProgressBar.setValue(percent)
            self.ui.trainingProgressBar.setToolTip(f"{current}/{total} epoch is completed")

            dice = train_stats.get("best_metric", 0)
            self.updateAccuracyBar(dice)
            return

        print("Training completed")
        self.ui.trainingProgressBar.setValue(100)
        self.timer.stop()
        self.timer = None
        self.ui.trainingProgressBar.setToolTip(f"Training: {status.get('status', 'DONE')}")

        self.ui.trainingButton.setEnabled(True)
        self.ui.stopTrainingButton.setEnabled(False)
        self.fetchInfo()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        file_ext = slicer.util.settingsValue("MONAILabel/fileExtension", self.file_ext)
        self.file_ext = file_ext if file_ext else self.file_ext

        # Update node selectors and sliders
        self.ui.inputSelector.clear()
        for v in self._volumeNodes:
            self.ui.inputSelector.addItem(v.GetName())
            self.ui.inputSelector.setToolTip(self.current_sample.get("name", "") if self.current_sample else "")
        if self._volumeNode:
            self.ui.inputSelector.setCurrentIndex(self.ui.inputSelector.findText(self._volumeNode.GetName()))
        self.ui.inputSelector.setEnabled(False)  # Allow only one active scene

        self.ui.uploadImageButton.setEnabled(False)
        if self.info and slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode") and self._volumeNode is None:
            self._volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            self.initSample({"id": self._volumeNode.GetName(), "session": True}, autosegment=False)
            self.ui.inputSelector.setEnabled(False)

        self.ui.uploadImageButton.setEnabled(self.current_sample and self.current_sample.get("session"))

        self.updateSelector(self.ui.segmentationModelSelector, ["segmentation"], "SegmentationModel", 0)
        self.updateSelector(self.ui.deepgrowModelSelector, ["deepgrow", "deepedit"], "DeepgrowModel", 0)
        self.updateSelector(self.ui.scribblesMethodSelector, ["scribbles"], "ScribblesMethod", 0)

        if self.models and [k for k, v in self.models.items() if v["type"] == "segmentation"]:
            self.ui.segmentationCollapsibleButton.collapsed = False
            self.ui.segmentationCollapsibleButton.show()
        else:
            self.ui.segmentationCollapsibleButton.hide()

        if self.models and [k for k, v in self.models.items() if v["type"] in ("deepgrow", "deepedit")]:
            self.ui.deepgrowCollapsibleButton.collapsed = False
            self.ui.deepgrowCollapsibleButton.show()
        else:
            self.ui.deepgrowCollapsibleButton.hide()

        if self.models and [k for k, v in self.models.items() if v["type"] == "scribbles"]:
            self.ui.scribblesCollapsibleButton.collapsed = False
            self.ui.scribblesCollapsibleButton.show()
        else:
            self.ui.scribblesCollapsibleButton.hide()

        if self.info.get("trainers", {}):
            self.ui.trainWidget.show()
        else:
            self.ui.trainWidget.hide()

        self.ignoreScribblesLabelChangeEvent = True
        self.ui.labelComboBox.clear()
        self.ui.scribLabelComboBox.clear()
        if self._segmentNode:
            segmentation = self._segmentNode.GetSegmentation()
            totalSegments = segmentation.GetNumberOfSegments()
            segmentIds = [segmentation.GetNthSegmentID(i) for i in range(totalSegments)]
            for idx, segmentId in enumerate(segmentIds):
                segment = segmentation.GetSegment(segmentId)
                label = segment.GetName()
                if label not in ["foreground_scribbles", "background_scribbles"]:
                    self.ui.labelComboBox.addItem(label)
                if label not in ["background", "foreground_scribbles", "background_scribbles"]:
                    self.ui.scribLabelComboBox.addItem(label)
        else:
            for label in self.info.get("labels", {}):
                self.ui.labelComboBox.addItem(label)
                if label != "background":
                    self.ui.scribLabelComboBox.addItem(label)

        currentLabel = self._parameterNode.GetParameter("CurrentLabel")
        idx = self.ui.labelComboBox.findText(currentLabel) if currentLabel else 0
        idx = 0 if idx < 0 < self.ui.labelComboBox.count else idx
        self.ui.labelComboBox.setCurrentIndex(idx)

        currentScribbleLabel = self._parameterNode.GetParameter("CurrentScribLabel")
        idx = self.ui.scribLabelComboBox.findText(currentScribbleLabel) if currentScribbleLabel else 0
        idx = 0 if idx < 0 < self.ui.scribLabelComboBox.count else idx
        self.ui.scribLabelComboBox.setCurrentIndex(idx)
        self.ignoreScribblesLabelChangeEvent = False

        self.ui.appComboBox.clear()
        self.ui.appComboBox.addItem(self.info.get("name", ""))

        datastore_stats = self.info.get("datastore", {})
        current = datastore_stats.get("completed", 0)
        total = datastore_stats.get("total", 0)
        self.ui.activeLearningProgressBar.setValue(current / max(total, 1) * 100)
        self.ui.activeLearningProgressBar.setToolTip(f"{current}/{total} samples are labeled")

        train_stats = self.info.get("train_stats", {})
        train_stats = next(iter(train_stats.values())) if train_stats else train_stats

        dice = train_stats.get("best_metric", 0)
        self.updateAccuracyBar(dice)

        self.ui.strategyBox.clear()
        for strategy in self.info.get("strategies", {}):
            self.ui.strategyBox.addItem(strategy)
        currentStrategy = self._parameterNode.GetParameter("CurrentStrategy")
        currentStrategy = currentStrategy if currentStrategy else self.state["CurrentStrategy"]
        self.ui.strategyBox.setCurrentIndex(self.ui.strategyBox.findText(currentStrategy) if currentStrategy else 0)

        self.ui.trainerBox.clear()
        trainers = self.info.get("trainers", {})
        if trainers:
            self.ui.trainerBox.addItem("ALL")
        for t in trainers:
            self.ui.trainerBox.addItem(t)
        currentTrainer = self._parameterNode.GetParameter("CurrentTrainer")
        currentTrainer = currentTrainer if currentTrainer else self.state["CurrentTrainer"]
        self.ui.trainerBox.setCurrentIndex(self.ui.trainerBox.findText(currentTrainer) if currentTrainer else 0)

        developer_mode = slicer.util.settingsValue("MONAILabel/developerMode", True, converter=slicer.util.toBool)
        self.ui.optionsCollapsibleButton.setVisible(developer_mode)

        # Enable/Disable
        self.ui.nextSampleButton.setEnabled(self.ui.strategyBox.count)

        is_training_running = True if self.info and self.isTrainingRunning() else False
        self.ui.trainingButton.setEnabled(self.info and not is_training_running and current)
        self.ui.stopTrainingButton.setEnabled(is_training_running)
        if is_training_running and self.timer is None:
            self.timer = qt.QTimer()
            self.timer.setInterval(5000)
            self.timer.connect("timeout()", self.monitorTraining)
            self.timer.start()

        self.ui.segmentationButton.setEnabled(
            self.ui.segmentationModelSelector.currentText and self._volumeNode is not None
        )
        self.ui.saveLabelButton.setEnabled(self._segmentNode is not None)
        self.ui.importLabelButton.setEnabled(self._segmentNode is not None)

        # Create empty markup point list node for deep grow +ve and -ve
        if self._segmentNode:
            if not self.dgPositivePointListNode:
                self.dgPositivePointListNode, self.dgPositivePointListNodeObservers = self.createPointListNode(
                    "P", self.onDeepGrowPointListNodeModified, [0.5, 1, 0.5]
                )
                self.ui.dgPositiveControlPointPlacementWidget.setCurrentNode(self.dgPositivePointListNode)
                self.ui.dgPositiveControlPointPlacementWidget.setPlaceModeEnabled(False)

            if not self.dgNegativePointListNode:
                self.dgNegativePointListNode, self.dgNegativePointListNodeObservers = self.createPointListNode(
                    "N", self.onDeepGrowPointListNodeModified, [0.5, 0.5, 1]
                )
                self.ui.dgNegativeControlPointPlacementWidget.setCurrentNode(self.dgNegativePointListNode)
                self.ui.dgNegativeControlPointPlacementWidget.setPlaceModeEnabled(False)

            self.ui.scribblesCollapsibleButton.setEnabled(self.ui.scribblesMethodSelector.count)
            self.ui.scribblesCollapsibleButton.collapsed = False

        self.ui.dgPositiveControlPointPlacementWidget.setEnabled(self.ui.deepgrowModelSelector.currentText)
        self.ui.dgNegativeControlPointPlacementWidget.setEnabled(self.ui.deepgrowModelSelector.currentText)

        self.deepedit_multi_label = False
        m = self.models.get(self.ui.deepgrowModelSelector.currentText) if self.models else None
        self.deepedit_multi_label = m and m.get("type") == "deepedit" and len(m.get("labels")) > 0

        if self.deepedit_multi_label:
            self.ui.dgLabelBackground.hide()
            self.ui.dgNegativeControlPointPlacementWidget.hide()
            self.ui.freezeUpdateCheckBox.show()
            self.ui.dgLabelForeground.setText("Landmarks:")
        else:
            self.ui.dgNegativeControlPointPlacementWidget.show()
            self.ui.freezeUpdateCheckBox.hide()
            self.ui.dgLabelForeground.setText("Foreground:")

        self.ui.dgUpdateCheckBox.setEnabled(self.ui.deepgrowModelSelector.currentText and self._segmentNode)
        self.ui.dgUpdateButton.setEnabled(self.ui.deepgrowModelSelector.currentText and self._segmentNode)

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        segmentationModelIndex = self.ui.segmentationModelSelector.currentIndex
        if segmentationModelIndex >= 0:
            segmentationModel = self.ui.segmentationModelSelector.itemText(segmentationModelIndex)
            self._parameterNode.SetParameter("SegmentationModel", segmentationModel)

        deepgrowModelIndex = self.ui.deepgrowModelSelector.currentIndex
        if deepgrowModelIndex >= 0:
            deepgrowModel = self.ui.deepgrowModelSelector.itemText(deepgrowModelIndex)
            self._parameterNode.SetParameter("DeepgrowModel", deepgrowModel)

        scribblesMethodIndex = self.ui.scribblesMethodSelector.currentIndex
        if scribblesMethodIndex >= 0:
            scribblesMethod = self.ui.scribblesMethodSelector.itemText(scribblesMethodIndex)
            self._parameterNode.SetParameter("ScribblesMethod", scribblesMethod)

        currentLabelIndex = self.ui.labelComboBox.currentIndex
        if currentLabelIndex >= 0:
            currentLabel = self.ui.labelComboBox.itemText(currentLabelIndex)
            self._parameterNode.SetParameter("CurrentLabel", currentLabel)

        currentScribLabelIndex = self.ui.scribLabelComboBox.currentIndex
        if currentScribLabelIndex >= 0:
            currentScribLabel = self.ui.scribLabelComboBox.itemText(currentScribLabelIndex)
            self._parameterNode.SetParameter("CurrentScribLabel", currentScribLabel)

        currentStrategyIndex = self.ui.strategyBox.currentIndex
        if currentStrategyIndex >= 0:
            currentStrategy = self.ui.strategyBox.itemText(currentStrategyIndex)
            self._parameterNode.SetParameter("CurrentStrategy", currentStrategy)

        currentTrainerIndex = self.ui.trainerBox.currentIndex
        if currentTrainerIndex >= 0:
            currentTrainer = self.ui.trainerBox.itemText(currentTrainerIndex)
            self._parameterNode.SetParameter("CurrentTrainer", currentTrainer)

        self._parameterNode.EndModify(wasModified)

    def updateSelector(self, selector, model_types, param, defaultIndex=0):
        wasSelectorBlocked = selector.blockSignals(True)
        selector.clear()

        for model_name, model in self.models.items():
            if model["type"] in model_types:
                selector.addItem(model_name)
                selector.setItemData(selector.count - 1, model["description"], qt.Qt.ToolTipRole)

        model = self._parameterNode.GetParameter(param)
        model = model if model else self.state.get(param, "")
        modelIndex = selector.findText(model)
        modelIndex = defaultIndex if modelIndex < 0 < selector.count else modelIndex
        selector.setCurrentIndex(modelIndex)

        try:
            modelInfo = self.models[model]
            selector.setToolTip(modelInfo["description"])
        except:
            selector.setToolTip("")
        selector.blockSignals(wasSelectorBlocked)

    def getSelectedOptionSection(self, index=-1):
        optionsSectionIndex = index if index >= 0 else self.ui.optionsSection.currentIndex
        optionsSectionIndex = optionsSectionIndex if optionsSectionIndex > 0 else 0
        optionsSection = self.ui.optionsSection.itemText(optionsSectionIndex)

        logging.info(f"Current Selection Options Section: {optionsSection}")
        mapping = {"infer": "models", "train": "trainers", "activelearning": "strategies", "scoring": "scoring"}

        return mapping.get(optionsSection)

    def getSelectedOptionName(self, index=-1):
        optionsNameIndex = index if index >= 0 else self.ui.optionsName.currentIndex
        optionsNameIndex = optionsNameIndex if optionsNameIndex > 0 else 0
        optionsName = self.ui.optionsName.itemText(optionsNameIndex)

        logging.info(f"Current Selection Options Name: {optionsName}")
        return optionsName

    def invalidateConfigTable(self, selection=-1, name=-1):
        section = self.getSelectedOptionSection(selection)
        name = self.getSelectedOptionName(name)
        if not section or not name:
            return

        mapping = {"infer": "models", "train": "trainers", "activelearning": "strategies", "scoring": "scoring"}
        section = mapping.get(section, section)
        for row in range(self.ui.configTable.rowCount):
            key = str(self.ui.configTable.item(row, 0).text())
            value = self.ui.configTable.item(row, 1)

            v = self.info.get(section, {}).get(name, {}).get("config", {}).get(key, {})
            if value is None:
                value = self.ui.configTable.cellWidget(row, 1)
                if isinstance(value, qt.QCheckBox):
                    value = True if value.checked else False
                else:
                    value = value.currentText
            else:
                value = str(value.text())

            if isinstance(v, bool):
                value = True if value else False
            elif isinstance(v, int):
                value = int(value) if value else 0
            elif isinstance(v, float):
                value = float(value) if value else 0.0
            elif isinstance(v, list):
                v.remove(value)
                v.insert(0, value)
                value = v

            logging.info(f"Invalidate:: {section} => {name} => {key} => {value} => {type(v)}")
            self.info.get(section, {}).get(name, {}).get("config", {})[key] = value

    def updateConfigTable(self, refresh=True):
        logging.info(f"updateConfigTable => refresh:{refresh}")
        section = self.getSelectedOptionSection()
        sectionConfig = self.info.get(section, {})
        if refresh:
            self.ui.optionsName.blockSignals(True)
            self.ui.optionsName.clear()
            for k in sectionConfig.keys():
                if sectionConfig[k].get("config"):
                    self.ui.optionsName.addItem(k)
            if self.ui.optionsName.count:
                self.ui.optionsName.setCurrentIndex(0)
            self.ui.optionsName.blockSignals(False)

        name = self.getSelectedOptionName()
        nameConfig = sectionConfig.get(name, {}).get("config", {})

        table = self.ui.configTable
        table.clear()
        headers = ["key", "value"]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setColumnWidth(0, 250)
        table.setRowCount(len(nameConfig))

        n = 0
        for key, val in nameConfig.items():
            item = qt.QTableWidgetItem(key)
            table.setItem(n, 0, item)
            item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)

            if isinstance(val, dict) or isinstance(val, list):
                combo = qt.QComboBox()
                for m, v in enumerate(val):
                    combo.addItem(v)
                combo.setCurrentIndex(0)
                table.setCellWidget(n, 1, combo)
            elif isinstance(val, bool):
                checkbox = qt.QCheckBox()
                checkbox.setChecked(val)
                table.setCellWidget(n, 1, checkbox)
            else:
                table.setItem(n, 1, qt.QTableWidgetItem(str(val) if val else ""))

            logging.info(f"{n} => {section} => {name} => {key} => {val}")
            n = n + 1

    def updateAccuracyBar(self, dice):
        self.ui.accuracyProgressBar.setValue(dice * 100)
        css = ["stop: 0 red"]
        if dice > 0.5:
            css.append(f"stop: {0.5 / dice} orange")
        if dice > 0.6:
            css.append(f"stop: {0.6 / dice} yellow")
        if dice > 0.7:
            css.append(f"stop: {0.7 / dice} lightgreen")
        if dice > 0.8:
            css.append(f"stop: {0.8 / dice} green")
        if dice > 0.9:
            css.append(f"stop: {0.9 / dice} darkgreen")
        self.ui.accuracyProgressBar.setStyleSheet(
            "QProgressBar {text-align: center;} "
            "QProgressBar::chunk {background-color: "
            "qlineargradient(x0: 0, x2: 1, " + ",".join(css) + ")}"
        )
        self.ui.accuracyProgressBar.setToolTip(f"Accuracy: {dice:.4f}")

    def getParamsFromConfig(self, section, name):
        self.invalidateConfigTable()

        mapping = {"infer": "models", "train": "trainers", "activelearning": "strategies", "scoring": "scoring"}
        section = mapping.get(section, section)
        sectionConfig = self.info.get(section, {})
        nameConfig = sectionConfig.get(name, {}).get("config", {})

        return {k: v[0] if isinstance(v, list) else v for k, v in nameConfig.items()}

    def onDeepGrowPointListNodeModified(self, observer, eventid):
        logging.debug("Deepgrow Point Event!!")

        if self.ignorePointListNodeAddEvent:
            return

        markupsNode = observer
        movingMarkupIndex = markupsNode.GetDisplayNode().GetActiveControlPoint()
        logging.debug(f"Markup point added; point ID = {movingMarkupIndex}")

        current_point = self.getControlPointXYZ(markupsNode, movingMarkupIndex)

        if not self.ui.dgUpdateCheckBox.checked:
            self.onClickDeepgrow(current_point, skip_infer=True)
            return

        self.onClickDeepgrow(current_point)

        self.ignorePointListNodeAddEvent = True
        self.onEditControlPoints(self.dgPositivePointListNode, "MONAILabel.ForegroundPoints")
        self.onEditControlPoints(self.dgNegativePointListNode, "MONAILabel.BackgroundPoints")
        self.ignorePointListNodeAddEvent = False

    def getControlPointsXYZ(self, pointListNode, name):
        v = self._volumeNode
        RasToIjkMatrix = vtk.vtkMatrix4x4()
        v.GetRASToIJKMatrix(RasToIjkMatrix)

        point_set = []
        n = pointListNode.GetNumberOfControlPoints()
        for i in range(n):
            coord = pointListNode.GetNthControlPointPosition(i)

            world = [0, 0, 0]
            pointListNode.GetNthControlPointPositionWorld(i, world)

            p_Ras = [coord[0], coord[1], coord[2], 1.0]
            p_Ijk = RasToIjkMatrix.MultiplyDoublePoint(p_Ras)
            p_Ijk = [round(i) for i in p_Ijk]

            logging.debug(f"RAS: {coord}; WORLD: {world}; IJK: {p_Ijk}")
            point_set.append(p_Ijk[0:3])

        logging.info(f"{name} => Current control points: {point_set}")
        return point_set

    def getControlPointXYZ(self, pointListNode, index):
        v = self._volumeNode
        RasToIjkMatrix = vtk.vtkMatrix4x4()
        v.GetRASToIJKMatrix(RasToIjkMatrix)

        coord = pointListNode.GetNthControlPointPosition(index)

        world = [0, 0, 0]
        pointListNode.GetNthControlPointPositionWorld(index, world)

        p_Ras = [coord[0], coord[1], coord[2], 1.0]
        p_Ijk = RasToIjkMatrix.MultiplyDoublePoint(p_Ras)
        p_Ijk = [round(i) for i in p_Ijk]

        logging.debug(f"RAS: {coord}; WORLD: {world}; IJK: {p_Ijk}")
        return p_Ijk[0:3]

    def onEditControlPoints(self, pointListNode, tagName):
        if pointListNode is None:
            return

        pointListNode.RemoveAllControlPoints()
        segmentId, segment = self.currentSegment()
        if segment and segmentId:
            v = self._volumeNode
            IjkToRasMatrix = vtk.vtkMatrix4x4()
            v.GetIJKToRASMatrix(IjkToRasMatrix)

            fPosStr = vtk.mutable("")
            segment.GetTag(tagName, fPosStr)
            pointset = str(fPosStr)
            logging.debug(f"{segmentId} => {segment.GetName()} Control points are: {pointset}")

            if fPosStr is not None and len(pointset) > 0:
                points = json.loads(pointset)
                for p in points:
                    p_Ijk = [p[0], p[1], p[2], 1.0]
                    p_Ras = IjkToRasMatrix.MultiplyDoublePoint(p_Ijk)
                    logging.debug(f"Add Control Point: {p_Ijk} => {p_Ras}")
                    pointListNode.AddControlPoint(p_Ras[0:3])

    def currentSegment(self):
        segmentation = self._segmentNode.GetSegmentation()
        segmentId = segmentation.GetSegmentIdBySegmentName(self.ui.labelComboBox.currentText)
        segment = segmentation.GetSegment(segmentId)

        logging.debug(f"Current SegmentID: {segmentId}; Segment: {segment}")
        return segmentId, segment

    def currentScribSegment(self):
        segmentation = self._segmentNode.GetSegmentation()
        segmentId = segmentation.GetSegmentIdBySegmentName(self.ui.scribLabelComboBox.currentText)
        segment = segmentation.GetSegment(segmentId)

        logging.debug(f"Current SegmentID: {segmentId}; Segment: {segment}")
        return segmentId, segment

    def onSelectLabel(self, caller=None, event=None):
        self.updateParameterNodeFromGUI(caller, event)

        self.ignorePointListNodeAddEvent = True
        self.onEditControlPoints(self.dgPositivePointListNode, "MONAILabel.ForegroundPoints")
        self.onEditControlPoints(self.dgNegativePointListNode, "MONAILabel.BackgroundPoints")
        self.ignorePointListNodeAddEvent = False

    def onSelectOptionsSection(self, index, caller=None, event=None):
        self.updateParameterNodeFromGUI(caller, event)
        logging.info(f"Options Section Selection Changed.... current:{index}; prev: {self.optionsSectionIndex}")

        self.invalidateConfigTable(self.optionsSectionIndex, self.optionsNameIndex)
        self.optionsSectionIndex = index
        self.optionsNameIndex = 0
        self.updateConfigTable()

    def onSelectOptionsName(self, index, caller=None, event=None):
        self.updateParameterNodeFromGUI(caller, event)
        logging.info(f"Options Name Selection Changed.... current:{index}; prev: {self.optionsNameIndex}")

        self.invalidateConfigTable(self.optionsSectionIndex, self.optionsNameIndex)
        self.optionsNameIndex = index
        self.updateConfigTable(refresh=False)

    def onSelectScribLabel(self, caller=None, event=None):
        if self.scribblesLayersPresent() and not self.ignoreScribblesLabelChangeEvent:
            if not slicer.util.confirmOkCancelDisplay(
                "This will clear current scribbles session.\n" "Are you sure to continue?"
            ):
                # undo changes to combobox
                currentScribLabel = self._parameterNode.GetParameter("CurrentScribLabel")
                logging.info(f"Cancel: reverting to original selection {currentScribLabel}")
                self.ignoreScribblesLabelChangeEvent = True
                self.ui.scribLabelComboBox.setCurrentIndex(self.ui.scribLabelComboBox.findText(currentScribLabel))
                self.ignoreScribblesLabelChangeEvent = False
                return
            self.onClearScribbles()

        self.updateParameterNodeFromGUI(caller, event)

    def icon(self, name="MONAILabel.png"):
        # It should not be necessary to modify this method
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", name)
        if os.path.exists(iconPath):
            return qt.QIcon(iconPath)
        return qt.QIcon()

    def updateServerSettings(self):
        self.logic.setServer(self.serverUrl())
        self.logic.setClientId(slicer.util.settingsValue("MONAILabel/clientId", "user-xyz"))
        self.saveServerUrl()

    def serverUrl(self):
        serverUrl = self.ui.serverComboBox.currentText
        if not serverUrl:
            serverUrl = "http://127.0.0.1:8000"
        return serverUrl.rstrip("/")

    def saveServerUrl(self):
        self.updateParameterNodeFromGUI()

        # Save selected server URL
        settings = qt.QSettings()
        serverUrl = self.ui.serverComboBox.currentText
        settings.setValue("MONAILabel/serverUrl", serverUrl)

        # Save current server URL to the top of history
        serverUrlHistory = settings.value("MONAILabel/serverUrlHistory")
        if serverUrlHistory:
            serverUrlHistory = serverUrlHistory.split(";")
        else:
            serverUrlHistory = []
        try:
            serverUrlHistory.remove(serverUrl)
        except ValueError:
            pass

        serverUrlHistory.insert(0, serverUrl)
        serverUrlHistory = serverUrlHistory[:10]  # keep up to first 10 elements
        settings.setValue("MONAILabel/serverUrlHistory", ";".join(serverUrlHistory))

        self.updateServerUrlGUIFromSettings()

    def onClickFetchInfo(self):
        self.fetchInfo()
        self.updateConfigTable()

    def fetchInfo(self, showInfo=False):
        if not self.logic:
            return

        start = time.time()
        try:
            self.updateServerSettings()
            info = self.logic.info()
            self.info = info
            if self.info.get("config"):
                slicer.util.errorDisplay(
                    "Please upgrade the monai server to latest version",
                    detailedText=traceback.format_exc(),
                )
                return
        except:
            slicer.util.errorDisplay(
                "Failed to fetch models from remote server. "
                "Make sure server address is correct and <server_uri>/info/ "
                "is accessible in browser",
                detailedText=traceback.format_exc(),
            )
            return

        self.models.clear()
        self.config = info.get("config", {})

        model_count = {}
        models = info.get("models", {})
        for k, v in models.items():
            model_type = v.get("type", "segmentation")
            model_count[model_type] = model_count.get(model_type, 0) + 1

            logging.debug(f"{k} = {model_type}")
            self.models[k] = v

        self.updateGUIFromParameterNode()

        msg = ""
        msg += "-----------------------------------------------------\t\n"
        msg += "Total Models Available: \t" + str(len(models)) + "\t\n"
        msg += "-----------------------------------------------------\t\n"
        for model_type in model_count.keys():
            msg += model_type.capitalize() + " Models: \t" + str(model_count[model_type]) + "\t\n"
        msg += "-----------------------------------------------------\t\n"

        if showInfo:
            qt.QMessageBox.information(slicer.util.mainWindow(), "MONAI Label", msg)
        logging.info(msg)
        logging.info(f"Time consumed by fetch info: {time.time() - start:3.1f}")

    def setProgressBarLabelText(self, label):
        if not self.progressBar:
            self.progressBar = slicer.util.createProgressDialog(windowTitle="Wait...", maximum=100)
        self.progressBar.labelText = label

    def reportProgress(self, progressPercentage):
        if not self.progressBar:
            self.progressBar = slicer.util.createProgressDialog(windowTitle="Wait...", maximum=100)
        self.progressBar.show()
        self.progressBar.activateWindow()
        self.progressBar.setValue(progressPercentage)
        slicer.app.processEvents()

    def onTraining(self):
        start = time.time()
        status = None
        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            self.updateServerSettings()

            model = self.ui.trainerBox.currentText
            if model == "ALL" and not slicer.util.confirmOkCancelDisplay(
                "This will trigger Training task for all models.  Are you sure to continue?"
            ):
                return

            model = model if model and model != "ALL" else None
            params = self.getParamsFromConfig("train", model)

            status = self.logic.train_start(model, params)

            self.ui.trainingProgressBar.setValue(1)
            self.ui.trainingProgressBar.setToolTip("Training: STARTED")

            time.sleep(1)
            self.updateGUIFromParameterNode()
        except:
            slicer.util.errorDisplay(
                "Failed to run training in MONAI Label Server", detailedText=traceback.format_exc()
            )
        finally:
            qt.QApplication.restoreOverrideCursor()

        if status:
            msg = "ID: {}\nStatus: {}\nStart Time: {}\n".format(
                status.get("id"),
                status.get("status"),
                status.get("start_ts"),
            )
            # slicer.util.infoDisplay(msg, detailedText=json.dumps(status, indent=2))
            logging.info(msg)

        logging.info(f"Time consumed by training: {time.time() - start:3.1f}")

    def onStopTraining(self):
        start = time.time()
        status = None
        if not slicer.util.confirmOkCancelDisplay(
            "This will kill/stop current Training task.  Are you sure to continue?"
        ):
            return

        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            self.updateServerSettings()
            status = self.logic.train_stop()
        except:
            slicer.util.errorDisplay("Failed to stop Training Task", detailedText=traceback.format_exc())
        finally:
            qt.QApplication.restoreOverrideCursor()

        if status:
            msg = "Status: {}\nStart Time: {}\nEnd Time: {}\nResult: {}".format(
                status.get("status"),
                status.get("start_ts"),
                status.get("end_ts"),
                status.get("result", status.get("details", [])[-1]),
            )
            # slicer.util.infoDisplay(msg, detailedText=json.dumps(status, indent=2))
            logging.info(msg)
        self.updateGUIFromParameterNode()

        logging.info(f"Time consumed by stop training: {time.time() - start:3.1f}")

    def isTrainingRunning(self, check_only=True):
        if not self.logic:
            return False
        self.updateServerSettings()
        return self.logic.train_status(check_only)

    def onNextSampleButton(self):
        if not self.logic:
            return

        if self._volumeNode or len(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")):
            if not slicer.util.confirmOkCancelDisplay(
                "This will close current scene.  Please make sure you have saved your current work.\n"
                "Are you sure to continue?"
            ):
                return
            self.onResetScribbles()
            slicer.mrmlScene.Clear(0)

        start = time.time()
        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

            self.updateServerSettings()
            strategy = self.ui.strategyBox.currentText
            if not strategy:
                slicer.util.errorDisplay("No Strategy Found/Selected\t")
                return

            sample = self.logic.next_sample(strategy, self.getParamsFromConfig("activelearning", strategy))
            logging.debug(sample)
            if not sample.get("id"):
                slicer.util.warningDisplay(
                    "Unlabled Samples/Images Not Found at server.  Instead you can load your own image."
                )
                return

            if self.samples.get(sample["id"]) is not None:
                self.current_sample = self.samples[sample["id"]]
                name = self.current_sample["VolumeNodeName"]
                index = self.ui.inputSelector.findText(name)
                self.ui.inputSelector.setCurrentIndex(index)
                return

            logging.info(sample)
            image_id = sample["id"]
            image_file = sample.get("path")
            image_name = sample.get("name", image_id)
            node_name = sample.get("PatientID", sample.get("name", image_id))
            checksum = sample.get("checksum")
            local_exists = image_file and os.path.exists(image_file)

            logging.info(f"Check if file exists/shared locally: {image_file} => {local_exists}")
            if local_exists:
                self._volumeNode = slicer.util.loadVolume(image_file)
                self._volumeNode.SetName(node_name)
            else:
                download_uri = f"{self.serverUrl()}/datastore/image?image={quote_plus(image_id)}"
                logging.info(download_uri)

                sampleDataLogic = SampleData.SampleDataLogic()
                self._volumeNode = sampleDataLogic.downloadFromURL(
                    nodeNames=node_name, fileNames=image_name, uris=download_uri, checksums=checksum
                )[0]

            if slicer.util.settingsValue("MONAILabel/originalLabel", True, converter=slicer.util.toBool):
                try:
                    datastore = self.logic.datastore()
                    labels = datastore["objects"][image_id]["labels"]["original"]["info"]["params"]["label_names"]
                    labels = labels.keys()
                    # ext = datastore['objects'][image_id]['labels']['original']['ext']
                    maskFile = self.logic.download_label(image_id, "original")
                    self.updateSegmentationMask(maskFile, list(labels))
                    print("Original label uploaded! ")

                except:
                    print("Original label not found ... ")

            self.initSample(sample)

        except:
            slicer.util.errorDisplay(
                "Failed to fetch Sample from MONAI Label Server", detailedText=traceback.format_exc()
            )
        finally:
            qt.QApplication.restoreOverrideCursor()

        self.updateGUIFromParameterNode()
        logging.info(f"Time consumed by next_sample: {time.time() - start:3.1f}")

    def initSample(self, sample, autosegment=True):
        sample["VolumeNodeName"] = self._volumeNode.GetName()
        self.current_sample = sample
        self.samples[sample["id"]] = sample
        self._volumeNodes.append(self._volumeNode)

        # Create Empty Segments for all labels for this node
        self.createSegmentNode()
        self.ui.embeddedSegmentEditorWidget.setSegmentationNode(self._segmentNode)
        self.ui.embeddedSegmentEditorWidget.setMasterVolumeNode(self._volumeNode)

        self.createScribblesROINode()
        self.ui.scribblesPlaceWidget.setCurrentNode(self._scribblesROINode)

        # check if user allows overlapping segments
        if slicer.util.settingsValue("MONAILabel/allowOverlappingSegments", False, converter=slicer.util.toBool):
            # set segment editor to allow overlaps
            self.logic.get_segment_editor_node().SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)

        if self.info.get("labels"):
            self.updateSegmentationMask(None, self.info.get("labels"))

        # Check if user wants to run auto-segmentation on new sample
        if autosegment and slicer.util.settingsValue(
            "MONAILabel/autoRunSegmentationOnNextSample", True, converter=slicer.util.toBool
        ):
            for label in self.info.get("labels", []):
                for name, model in self.models.items():
                    if label in model.get("labels", []):
                        qt.QApplication.restoreOverrideCursor()
                        self.ui.segmentationModelSelector.currentText = name
                        self.onClickSegmentation()
                        return

    def getPermissionForImageDataUpload(self):
        return slicer.util.confirmOkCancelDisplay(
            "Master volume - without any additional patient information -"
            " will be sent to remote data processing server: {}.\n\n"
            "Click 'OK' to proceed with the segmentation.\n"
            "Click 'Cancel' to not upload any data and cancel segmentation.\n".format(self.serverUrl()),
            dontShowAgainSettingsKey="MONAILabel/showImageDataSendWarning",
        )

    def onUploadImage(self, init_sample=True, session=False):
        volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        image_id = volumeNode.GetName()

        if not self.getPermissionForImageDataUpload():
            return False

        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            in_file = tempfile.NamedTemporaryFile(suffix=self.file_ext, dir=self.tmpdir).name
            self.reportProgress(5)

            start = time.time()
            slicer.util.saveNode(volumeNode, in_file)
            logging.info(f"Saved Input Node into {in_file} in {time.time() - start:3.1f}s")
            self.reportProgress(30)

            if session:
                self.current_sample["session_id"] = self.logic.create_session(in_file)["session_id"]
            else:
                self.logic.upload_image(in_file, image_id)
                self.current_sample["session"] = False
            self.reportProgress(100)

            self._volumeNode = volumeNode
            if init_sample:
                self.initSample({"id": image_id}, autosegment=False)
            qt.QApplication.restoreOverrideCursor()

            self.updateGUIFromParameterNode()
            return True
        except:
            self.reportProgress(100)
            qt.QApplication.restoreOverrideCursor()
            if session:
                slicer.util.errorDisplay(
                    "Server Error:: Session creation Failed\nPlease upgrade to latest monailable version (> 0.2.0)",
                    detailedText=traceback.format_exc(),
                )
                self.current_sample["session"] = None
            else:
                slicer.util.errorDisplay("Failed to upload volume to Server", detailedText=traceback.format_exc())
            return False

    def onImportLabel(self):
        if not self.ui.labelPathLineEdit.currentPath or not os.path.exists(self.ui.labelPathLineEdit.currentPath):
            slicer.util.warningDisplay("Label File not selected")
            return

        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            self.updateSegmentationMask(self.ui.labelPathLineEdit.currentPath, self.info["labels"])
            qt.QApplication.restoreOverrideCursor()
        except:
            qt.QApplication.restoreOverrideCursor()
            slicer.util.errorDisplay("Failed to import label", detailedText=traceback.format_exc())

    def onSaveLabel(self):
        start = time.time()
        labelmapVolumeNode = None
        result = None
        self.onResetScribbles()

        if self.current_sample.get("session"):
            if not self.onUploadImage(init_sample=False):
                return

        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            segmentationNode = self._segmentNode
            segmentation = segmentationNode.GetSegmentation()
            totalSegments = segmentation.GetNumberOfSegments()
            segmentIds = [segmentation.GetNthSegmentID(i) for i in range(totalSegments)]

            # remove background and scribbles labels
            label_info = []
            save_segment_ids = vtk.vtkStringArray()
            for idx, segmentId in enumerate(segmentIds):
                segment = segmentation.GetSegment(segmentId)
                if segment.GetName() in ["background", "foreground_scribbles", "background_scribbles"]:
                    logging.info(f"Removing segment {segmentId}: {segment.GetName()}")
                    continue

                save_segment_ids.InsertNextValue(segmentId)
                label_info.append({"name": segment.GetName(), "idx": idx + 1})
                # label_info.append({"color": segment.GetColor()})

            # export labelmap
            labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segmentationNode, save_segment_ids, labelmapVolumeNode, self._volumeNode
            )

            label_in = tempfile.NamedTemporaryFile(suffix=self.file_ext, dir=self.tmpdir).name
            self.reportProgress(5)

            if (
                slicer.util.settingsValue("MONAILabel/allowOverlappingSegments", True, converter=slicer.util.toBool)
                and slicer.util.settingsValue("MONAILabel/fileExtension", self.file_ext) == ".seg.nrrd"
            ):
                slicer.util.saveNode(segmentationNode, label_in)
            else:
                slicer.util.saveNode(labelmapVolumeNode, label_in)
            self.reportProgress(30)

            self.updateServerSettings()
            result = self.logic.save_label(self.current_sample["id"], label_in, {"label_info": label_info})
            self.fetchInfo()

            if slicer.util.settingsValue("MONAILabel/autoUpdateModelV2", False, converter=slicer.util.toBool):
                try:
                    if self.isTrainingRunning(check_only=True):
                        self.logic.train_stop()
                except:
                    logging.info("Failed to stop training; or already stopped")
                self.onTraining()
        except:
            slicer.util.errorDisplay("Failed to save Label to MONAI Label Server", detailedText=traceback.format_exc())
        finally:
            qt.QApplication.restoreOverrideCursor()
            self.reportProgress(100)

            if labelmapVolumeNode:
                slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
            if result:
                slicer.util.infoDisplay(
                    "Label-Mask saved into MONAI Label Server\t\t", detailedText=json.dumps(result, indent=2)
                )

                if slicer.util.settingsValue("MONAILabel/autoFetchNextSample", False, converter=slicer.util.toBool):
                    slicer.mrmlScene.Clear(0)
                    self.onNextSampleButton()

        logging.info(f"Time consumed by save label: {time.time() - start:3.1f}")

    def getSessionId(self):
        session_id = None
        if self.current_sample.get("session", False):
            session_id = self.current_sample.get("session_id")
            if not session_id or not self.logic.get_session(session_id):
                self.onUploadImage(init_sample=False, session=True)
                session_id = self.current_sample["session_id"]
        return session_id

    def onClickSegmentation(self):
        if not self.current_sample:
            return

        start = time.time()
        result_file = None
        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

            self.updateServerSettings()

            model = self.ui.segmentationModelSelector.currentText
            image_file = self.current_sample["id"]
            params = self.getParamsFromConfig("infer", model)

            result_file, params = self.logic.infer(model, image_file, params, session_id=self.getSessionId())
            print(f"Result Params for Segmentation: {params}")

            labels = (
                params.get("label_names") if params and params.get("label_names") else self.models[model].get("labels")
            )
            if labels and isinstance(labels, dict):
                labels = [k for k, _ in sorted(labels.items(), key=lambda item: item[1])]
            self.updateSegmentationMask(result_file, labels)
        except:
            slicer.util.errorDisplay(
                "Failed to run inference in MONAI Label Server", detailedText=traceback.format_exc()
            )
        finally:
            qt.QApplication.restoreOverrideCursor()
            if result_file and os.path.exists(result_file):
                os.unlink(result_file)

        self.updateGUIFromParameterNode()
        logging.info(f"Time consumed by segmentation: {time.time() - start:3.1f}")

    def onUpdateDeepgrow(self):
        self.onClickDeepgrow(None)

    def onClickDeepgrow(self, current_point, skip_infer=False):
        model = self.ui.deepgrowModelSelector.currentText
        if not model:
            slicer.util.warningDisplay("Please select a deepgrow model")
            return

        _, segment = self.currentSegment()
        if not segment:
            slicer.util.warningDisplay("Please add the required label to run deepgrow")
            return

        foreground_all = self.getControlPointsXYZ(self.dgPositivePointListNode, "foreground")
        background_all = self.getControlPointsXYZ(self.dgNegativePointListNode, "background")

        segment.SetTag("MONAILabel.ForegroundPoints", json.dumps(foreground_all))
        segment.SetTag("MONAILabel.BackgroundPoints", json.dumps(background_all))
        if skip_infer:
            return

        # use model info "deepgrow" to determine
        deepgrow_3d = False if self.models[model].get("dimension", 3) == 2 else True
        print(f"Is DeepGrow 3D: {deepgrow_3d}")
        start = time.time()

        label = segment.GetName()
        operationDescription = f"Run Deepgrow for segment: {label}; model: {model}; 3d {deepgrow_3d}"
        logging.debug(operationDescription)

        if not current_point:
            if not foreground_all and not deepgrow_3d:
                slicer.util.warningDisplay(operationDescription + " - points not added")
                return
            current_point = foreground_all[-1] if foreground_all else background_all[-1] if background_all else None

        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

            sliceIndex = None
            if self.deepedit_multi_label:
                params = {}
                segmentation = self._segmentNode.GetSegmentation()
                for name in self.info.get("labels", []):
                    points = []
                    segmentId = segmentation.GetSegmentIdBySegmentName(name)
                    segment = segmentation.GetSegment(segmentId) if segmentId else None
                    if segment:
                        fPosStr = vtk.mutable("")
                        segment.GetTag("MONAILabel.ForegroundPoints", fPosStr)
                        pointset = str(fPosStr)
                        print(f"{segmentId} => {name} Control points are: {pointset}")
                        if fPosStr is not None and len(pointset) > 0:
                            points = json.loads(pointset)

                    params[name] = points
                params["label"] = label
                labels = None
            else:
                sliceIndex = current_point[2] if current_point else None
                print(f"Slice Index: {sliceIndex}")

                if deepgrow_3d or not sliceIndex:
                    foreground = foreground_all
                    background = background_all
                else:
                    foreground = [x for x in foreground_all if x[2] == sliceIndex]
                    background = [x for x in background_all if x[2] == sliceIndex]

                logging.debug(f"Foreground: {foreground}")
                logging.debug(f"Background: {background}")
                logging.debug(f"Current point: {current_point}")

                params = {
                    "label": label,
                    "foreground": foreground,
                    "background": background,
                }
                labels = [label]

            params["label"] = label
            params.update(self.getParamsFromConfig("infer", model))
            print(f"Request Params for Deepgrow/Deepedit: {params}")

            image_file = self.current_sample["id"]
            result_file, params = self.logic.infer(model, image_file, params, session_id=self.getSessionId())
            print(f"Result Params for Deepgrow/Deepedit: {params}")
            if labels is None:
                labels = (
                    params.get("label_names")
                    if params and params.get("label_names")
                    else self.models[model].get("labels")
                )
                if labels and isinstance(labels, dict):
                    labels = [k for k, _ in sorted(labels.items(), key=lambda item: item[1])]

            freeze = label if self.ui.freezeUpdateCheckBox.checked else None
            self.updateSegmentationMask(result_file, labels, None if deepgrow_3d else sliceIndex, freeze=freeze)
        except:
            logging.exception("Unknown Exception")
            slicer.util.errorDisplay(operationDescription + " - unexpected error.", detailedText=traceback.format_exc())
        finally:
            qt.QApplication.restoreOverrideCursor()

        self.updateGUIFromParameterNode()
        logging.info(f"Time consumed by Deepgrow: {time.time() - start:3.1f}")

    def createCursor(self, widget):
        return slicer.util.mainWindow().cursor

    def createSegmentNode(self):
        if self._volumeNode is None:
            return
        if self._segmentNode is None:
            name = "segmentation_" + self._volumeNode.GetName()
            self._segmentNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            self._segmentNode.SetReferenceImageGeometryParameterFromVolumeNode(self._volumeNode)
            self._segmentNode.SetName(name)

    def createScribblesROINode(self):
        if self._volumeNode is None:
            return
        if self._scribblesROINode is None:
            scribblesROINode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
            scribblesROINode.SetName("Scribbles ROI")
            scribblesROINode.CreateDefaultDisplayNodes()
            scribblesROINode.GetDisplayNode().SetFillOpacity(0.4)
            scribblesROINode.GetDisplayNode().SetSelectedColor(1, 1, 1)
            scribblesROINode.GetDisplayNode().SetColor(1, 1, 1)
            scribblesROINode.GetDisplayNode().SetActiveColor(1, 1, 1)
            self._scribblesROINode = scribblesROINode

    def getLabelColor(self, name):
        color = GenericAnatomyColors.get(name.lower())
        return [c / 255.0 for c in color] if color else None

    def updateSegmentationMask(self, in_file, labels, sliceIndex=None, freeze=None):
        # TODO:: Add ROI Node (for Bounding Box if provided in the result)
        start = time.time()
        logging.debug(f"Update Segmentation Mask from: {in_file}")
        if in_file and not os.path.exists(in_file):
            return False

        segmentationNode = self._segmentNode
        segmentation = segmentationNode.GetSegmentation()

        if in_file is None:
            for label in labels:
                if not segmentation.GetSegmentIdBySegmentName(label):
                    segmentation.AddEmptySegment(label, label, self.getLabelColor(label))
            return True

        if in_file.endswith(".seg.nrrd") and self.file_ext == ".seg.nrrd":
            source_node = slicer.modules.segmentations.logic().LoadSegmentationFromFile(in_file, False)
            destination_node = segmentationNode
            destination_segmentations = destination_node.GetSegmentation()
            source_segmentations = source_node.GetSegmentation()

            destination_segmentations.DeepCopy(source_segmentations)

            if self._volumeNode:
                destination_node.SetReferenceImageGeometryParameterFromVolumeNode(self._volumeNode)

            slicer.mrmlScene.RemoveNode(source_node)
        else:
            labels = [label for label in labels if label != "background"]
            logging.info(f"Update Segmentation Mask using Labels: {labels}")

            # segmentId, segment = self.currentSegment()
            labelImage = sitk.ReadImage(in_file)
            labelmapVolumeNode = sitkUtils.PushVolumeToSlicer(labelImage, None, className="vtkMRMLLabelMapVolumeNode")
            logging.info(f"Time consumed by Import LabelMask: {time.time() - start:3.1f}")

            freeze = [freeze] if freeze and isinstance(freeze, str) else freeze
            logging.info(f"Import only Freezed label: {freeze}")

            if sliceIndex is None and not freeze:
                # List of segments to import
                segmentIds = vtk.vtkStringArray()
                for label in labels:
                    segmentIds.InsertNextValue(label)

                # faster import (based on selected segmentIds)
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                    labelmapVolumeNode, segmentationNode, segmentIds
                )
                slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
            else:
                existingCount = segmentation.GetNumberOfSegments()
                existing_label_ids = {}
                for label in labels:
                    id = segmentation.GetSegmentIdBySegmentName(label)
                    if id:
                        existing_label_ids[label] = id

                # slower import (import all - use only when you have to update one particular slice for 2D)
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                    labelmapVolumeNode, segmentationNode
                )
                slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

                addedCount = segmentation.GetNumberOfSegments() - existingCount
                addedSegmentIds = [segmentation.GetNthSegmentID(existingCount + i) for i in range(addedCount)]

                self.ui.embeddedSegmentEditorWidget.setSegmentationNode(segmentationNode)
                self.ui.embeddedSegmentEditorWidget.setMasterVolumeNode(self._volumeNode)

                for i, segmentId in enumerate(addedSegmentIds):
                    label = labels[i] if i < len(labels) else f"unknown {i}"
                    if freeze and label not in freeze:
                        logging.info(f"Discard label update for: {label}")
                    else:
                        segment = segmentation.GetSegment(segmentId)
                        logging.info(f"select segmentation with id: {segmentId} => {segment.GetName()} => {label}")
                        if label in existing_label_ids:
                            l_start = time.time()
                            label_id = existing_label_ids[label]

                            self.ui.embeddedSegmentEditorWidget.setCurrentSegmentID(label_id)
                            effect = self.ui.embeddedSegmentEditorWidget.effectByName("Logical operators")

                            if sliceIndex is not None:
                                selectedSegmentLabelmap = effect.selectedSegmentLabelmap()
                                dims = selectedSegmentLabelmap.GetDimensions()
                                for x in range(dims[0]):
                                    for y in range(dims[1]):
                                        selectedSegmentLabelmap.SetScalarComponentFromDouble(x, y, sliceIndex, 0, 0)

                                logging.info(f"{label} - Time to Clean the slice: {time.time() - l_start:3.1f}")

                            l_start = time.time()
                            newLabelmap = slicer.vtkOrientedImageData()
                            segmentationNode.GetBinaryLabelmapRepresentation(segmentId, newLabelmap)
                            op = (
                                slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeSet
                                if sliceIndex is None
                                else slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeAdd
                            )
                            effect.modifySelectedSegmentByLabelmap(newLabelmap, op)
                            logging.info(f"{label} - Time to Update the segment: {time.time() - l_start:3.1f}")

                    segmentationNode.RemoveSegment(segmentId)
                    logging.info(f"Time consumed until Import Segment => {label}: {time.time() - start:3.1f}")

        if slicer.util.settingsValue("MONAILabel/showSegmentsIn3D", False, converter=slicer.util.toBool):
            self.showSegmentationsIn3D()

        logging.info(f"Time consumed by updateSegmentationMask: {time.time() - start:3.1f}")
        return True

    def showSegmentationsIn3D(self):
        # add closed surface representation
        if self._segmentNode:
            self._segmentNode.CreateClosedSurfaceRepresentation()
        view = slicer.app.layoutManager().threeDWidget(0).threeDView()
        view.resetFocalPoint()

    def updateServerUrlGUIFromSettings(self):
        # Save current server URL to the top of history
        settings = qt.QSettings()
        serverUrlHistory = settings.value("MONAILabel/serverUrlHistory")

        wasBlocked = self.ui.serverComboBox.blockSignals(True)
        self.ui.serverComboBox.clear()
        if serverUrlHistory:
            self.ui.serverComboBox.addItems(serverUrlHistory.split(";"))
        self.ui.serverComboBox.setCurrentText(settings.value("MONAILabel/serverUrl"))
        self.ui.serverComboBox.blockSignals(wasBlocked)

    def createPointListNode(self, name, onMarkupNodeModified, color):
        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsDisplayNode")
        displayNode.SetTextScale(0)
        displayNode.SetSelectedColor(color)

        pointListNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        pointListNode.SetName(name)
        pointListNode.SetAndObserveDisplayNodeID(displayNode.GetID())

        pointListNodeObservers = []
        self.addPointListNodeObserver(pointListNode, onMarkupNodeModified)
        return pointListNode, pointListNodeObservers

    def removePointListNodeObservers(self, pointListNode, pointListNodeObservers):
        if pointListNode and pointListNodeObservers:
            for observer in pointListNodeObservers:
                pointListNode.RemoveObserver(observer)

    def addPointListNodeObserver(self, pointListNode, onMarkupNodeModified):
        pointListNodeObservers = []
        if pointListNode:
            eventIds = [slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent]
            for eventId in eventIds:
                pointListNodeObservers.append(pointListNode.AddObserver(eventId, onMarkupNodeModified))
        return pointListNodeObservers

    def scribblesLayersPresent(self):
        scribbles_exist = False
        if self._segmentNode is not None:
            segmentationNode = self._segmentNode
            segmentation = segmentationNode.GetSegmentation()
            numSegments = segmentation.GetNumberOfSegments()
            segmentIds = [segmentation.GetNthSegmentID(i) for i in range(numSegments)]
            scribbles_exist = sum(int("scribbles" in sid) for sid in segmentIds) > 0
        return scribbles_exist

    def ensureScribblesLayersPresent(self):
        if (not self._segmentNode) or self.scribblesLayersPresent():
            return

        # add background, layer index = -2 [2], color = red
        self._segmentNode.GetSegmentation().AddEmptySegment(
            "background_scribbles", "background_scribbles", [1.0, 0.0, 0.0]
        )

        # add foreground, layer index = -1 [3], color = green
        self._segmentNode.GetSegmentation().AddEmptySegment(
            "foreground_scribbles", "foreground_scribbles", [0.0, 1.0, 0.0]
        )

        # change segmentation display properties to "see through" the scribbles
        # further explanation at:
        # https://apidocs.slicer.org/master/classvtkMRMLSegmentationDisplayNode.html
        segmentationDisplayNode = self._segmentNode.GetDisplayNode()

        # background
        opacity = 0.2
        segmentationDisplayNode.SetSegmentOpacity2DFill("background_scribbles", opacity)
        segmentationDisplayNode.SetSegmentOpacity2DOutline("background_scribbles", opacity)

        # foreground
        segmentationDisplayNode.SetSegmentOpacity2DFill("foreground_scribbles", opacity)
        segmentationDisplayNode.SetSegmentOpacity2DOutline("foreground_scribbles", opacity)

    def onUpdateScribbles(self):
        logging.info("Scribbles update event")
        scribblesMethod = self.ui.scribblesMethodSelector.currentText
        scribbles_in = None
        result_file = None

        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

            # get scribbles + label
            segmentationNode = self._segmentNode
            labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            save_segment_ids = vtk.vtkStringArray()
            segmentationNode.GetSegmentation().GetSegmentIDs(save_segment_ids)
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segmentationNode, save_segment_ids, labelmapVolumeNode, self._volumeNode
            )
            segmentation = segmentationNode.GetSegmentation()
            totalSegments = segmentation.GetNumberOfSegments()
            segmentIds = [segmentation.GetNthSegmentID(i) for i in range(totalSegments)]

            label_info = []
            for idx, segmentId in enumerate(segmentIds):
                segment = segmentation.GetSegment(segmentId)
                label_info.append({"name": segment.GetName(), "id": idx + 1})

            scribbles_in = tempfile.NamedTemporaryFile(suffix=self.file_ext, dir=self.tmpdir).name
            self.reportProgress(5)

            # save scribbles + label to file
            slicer.util.saveNode(labelmapVolumeNode, scribbles_in)
            slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
            self.reportProgress(30)
            self.updateServerSettings()
            self.reportProgress(60)

            # try to get roi if placed
            roiNode = self.ui.scribblesPlaceWidget.currentNode()
            selected_roi = []
            if roiNode and roiNode.GetControlPointPlacementComplete():
                selected_roi = self.getROIPointsXYZ(roiNode)

            # send scribbles + label to server along with selected scribbles method
            params = self.getParamsFromConfig("infer", scribblesMethod)
            params.update({"roi": selected_roi})
            params.update({"label_info": label_info})
            _, segment = self.currentScribSegment()
            selected_label_name = segment.GetName()
            params.update({"selected_label_name": selected_label_name})

            image_file = self.current_sample["id"]
            result_file, params = self.logic.infer(
                scribblesMethod, image_file, params, scribbles_in, session_id=self.getSessionId()
            )

            # display result from server
            self.reportProgress(90)
            self.updateSegmentationMask(result_file, [selected_label_name])
        except:
            slicer.util.errorDisplay(
                f"Failed to post process label on MONAI Label Server using {scribblesMethod}",
                detailedText=traceback.format_exc(),
            )
        finally:
            qt.QApplication.restoreOverrideCursor()
            self.reportProgress(100)

            # clear all temporary files
            if scribbles_in and os.path.exists(scribbles_in):
                os.unlink(scribbles_in)

            if result_file and os.path.exists(result_file):
                os.unlink(result_file)

    def getROIPointsXYZ(self, roiNode):
        if roiNode is None:
            return []

        v = self._volumeNode
        RasToIjkMatrix = vtk.vtkMatrix4x4()
        v.GetRASToIJKMatrix(RasToIjkMatrix)

        roi_points_ras = [0.0] * 6
        center = [0] * 3
        roiNode.GetCenter(center)
        roi_points_ras = [(x - s / 2, x + s / 2) for x, s in zip(center, roiNode.GetSize())]
        roi_points_ras = [item for sublist in roi_points_ras for item in sublist]

        min_points_ras = [roi_points_ras[0], roi_points_ras[2], roi_points_ras[4], 1.0]
        max_points_ras = [roi_points_ras[0 + 1], roi_points_ras[2 + 1], roi_points_ras[4 + 1], 1.0]

        min_points_ijk = RasToIjkMatrix.MultiplyDoublePoint(min_points_ras)
        max_points_ijk = RasToIjkMatrix.MultiplyDoublePoint(max_points_ras)

        min_points_ijk = [round(i) for i in min_points_ijk]
        max_points_ijk = [round(i) for i in max_points_ijk]

        roi_points_ijk = [val for pair in zip(min_points_ijk[0:3], max_points_ijk[0:3]) for val in pair]
        logging.debug(f"RAS: {roi_points_ras}; IJK: {roi_points_ijk}")
        # print("RAS: {}; IJK: {}".format(roi_points_ras, roi_points_ijk))

        return roi_points_ijk

    def onClearScribblesSegmentNodes(self):
        # more explanation on this at:
        # https://discourse.slicer.org/t/how-to-clear-segmentation/7433/4
        # clear "scribbles" segment before saving the label
        if not self._segmentNode:
            return

        segmentation = self._segmentNode
        num_segments = segmentation.GetSegmentation().GetNumberOfSegments()
        for i in range(num_segments):
            segmentId = segmentation.GetSegmentation().GetNthSegmentID(i)
            if "scribbles" in segmentId:
                logging.info(f"clearing {segmentId}")
                labelMapRep = slicer.vtkOrientedImageData()
                segmentation.GetBinaryLabelmapRepresentation(segmentId, labelMapRep)
                vtkSegmentationCore.vtkOrientedImageDataResample.FillImage(labelMapRep, 0, labelMapRep.GetExtent())
                slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
                    labelMapRep, segmentation, segmentId, slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE
                )

        # refresh segmentation view to clear scribbles segmentations
        # help from: https://discourse.slicer.org/t/refresh-volume-rendering/11847/6
        segmentation.SetDisplayVisibility(False)
        segmentation.SetDisplayVisibility(True)

    def resetScribblesROI(self):
        if self._scribblesROINode:
            self._scribblesROINode.RemoveAllControlPoints()

    def onClearScribbles(self):
        # for clearing scribbles and resetting tools to default
        # remove "scribbles" segments from label
        self.onClearScribblesSegmentNodes()

        self.resetScribblesROI()

        self.ui.paintScribblesButton.setChecked(True)
        self.ui.eraseScribblesButton.setChecked(False)

        self.ui.scribblesSelector.setCurrentIndex(0)

    def onResetScribbles(self):

        # reset scribbles mode
        self.scribblesMode = None

        # remove "scribbles" segments from label
        self.onClearScribblesSegmentNodes()

        self.ui.paintScribblesButton.setChecked(False)
        self.ui.eraseScribblesButton.setChecked(False)

        self.ui.scribblesSelector.setCurrentIndex(0)
        self.ignoreScribblesLabelChangeEvent = True
        self.ui.scribLabelComboBox.setCurrentIndex(0)
        self.ignoreScribblesLabelChangeEvent = False

    def updateScribToolLayerFromMode(self):
        if not self._segmentNode:
            return

        logging.info(f"Scribbles mode {self.scribblesMode} ")

        if self.scribblesMode is None:
            self.changeScribblesMode(tool="Paint", layer="foreground_scribbles")
            self.updateScribToolLayerFromMode()

        # update tool/layer select for scribblesEditorWidget
        tool, layer = self.getToolAndLayerFromScribblesMode()
        if self.scribblesMode is not None:
            self.ensureScribblesLayersPresent()

            # adding new scribbles can overwrite a new one-hot vector, hence erase any existing
            # labels - this is not a desired behaviour hence we swith to overlay mode that enables drawing
            # scribbles without changing existing labels. Further explanation at:
            # https://discourse.slicer.org/t/how-can-i-set-masking-settings-on-a-segment-editor-effect-in-python/4406/7
            self.logic.get_segment_editor_node().SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)

            self.ui.embeddedSegmentEditorWidget.setActiveEffectByName(tool)
            self.ui.embeddedSegmentEditorWidget.setCurrentSegmentID(layer)

        # update brush type from checkbox
        if tool in ("Paint", "Erase"):
            is3dbrush = self.ui.brush3dCheckbox.checkState()
            self.on3dBrushCheckbox(state=is3dbrush)

            # update brush size from slider
            brushSize = self.ui.brushSizeSlider.value
            self.updateBrushSize(value=brushSize)

    def getToolAndLayerFromScribblesMode(self):
        if self.scribblesMode is not None:
            return self.scribblesMode.split("+")
        else:
            # default modes
            return "Paint", "foreground_scribbles"

    def changeScribblesMode(self, tool=None, layer=None):
        ctool, clayer = self.getToolAndLayerFromScribblesMode()

        ctool = tool if tool is not None else ctool
        clayer = layer if layer is not None else clayer

        self.scribblesMode = "+".join([ctool, clayer])

    def onPaintScribbles(self):
        if not self._segmentNode:
            return

        if self.ui.eraseScribblesButton.checked:
            self.ui.eraseScribblesButton.setChecked(False)

        self.changeScribblesMode(tool="Paint" if self.ui.paintScribblesButton.checked else "None")
        self.updateScribToolLayerFromMode()

    def onEraseScribbles(self):
        if not self._segmentNode:
            return

        if self.ui.paintScribblesButton.checked:
            self.ui.paintScribblesButton.setChecked(False)

        self.changeScribblesMode(tool="Erase" if self.ui.eraseScribblesButton.checked else "None")
        self.updateScribToolLayerFromMode()

    def onSelectScribblesLabel(self):
        if not self._segmentNode:
            return

        index = self.ui.scribblesSelector.currentIndex
        index = 0 if index < 0 else index
        selected = self.ui.scribblesSelector.itemText(index)

        layer = "foreground_scribbles" if selected == "Foreground" else "background_scribbles"
        self.changeScribblesMode(layer=layer)
        self.updateScribToolLayerFromMode()

    def on3dBrushCheckbox(self, state):
        logging.info(f"3D brush update {state}")
        # enable scribbles in 3d using a sphere brush
        effect = self.ui.embeddedSegmentEditorWidget.effectByName("Paint")
        effect.setParameter("BrushSphere", state)

    def updateBrushSize(self, value):
        logging.info(f"brush size update {value}")
        effect = self.ui.embeddedSegmentEditorWidget.effectByName("Paint")
        effect.setParameter("BrushAbsoluteDiameter", value)


class MONAILabelLogic(ScriptedLoadableModuleLogic):
    def __init__(self, tmpdir=None, server_url=None, progress_callback=None, client_id=None):
        ScriptedLoadableModuleLogic.__init__(self)

        self.server_url = server_url
        self.tmpdir = slicer.util.tempDirectory("slicer-monai-label") if tmpdir is None else tmpdir
        self.client_id = client_id

        self.volumeToSessions = dict()
        self.progress_callback = progress_callback

    def setDefaultParameters(self, parameterNode):
        if not parameterNode.GetParameter("SegmentationModel"):
            parameterNode.SetParameter("SegmentationModel", "")
        if not parameterNode.GetParameter("DeepgrowModel"):
            parameterNode.SetParameter("DeepgrowModel", "")
        if not parameterNode.GetParameter("ScribblesMethod"):
            parameterNode.SetParameter("ScribblesMethod", "")

    def __del__(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def setServer(self, server_url=None):
        self.server_url = server_url if server_url else "http://127.0.0.1:8000"

    def setClientId(self, client_id):
        self.client_id = client_id if client_id else "user-xyz"

    def setProgressCallback(self, progress_callback=None):
        self.progress_callback = progress_callback

    def reportProgress(self, progress):
        if self.progress_callback:
            self.progress_callback(progress)

    def get_segment_editor_node(self):
        # Use the Segment Editor module's parameter node for the embedded segment editor widget.
        # This ensures that if the user switches to the Segment Editor then the selected
        # segmentation node, volume node, etc. are the same.
        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if segmentEditorNode is None:
            segmentEditorNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorNode.UnRegister(None)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        return segmentEditorNode

    def info(self):
        return MONAILabelClient(self.server_url, self.tmpdir, self.client_id).info()

    def datastore(self):
        return MONAILabelClient(self.server_url, self.tmpdir, self.client_id).datastore()

    def download_label(self, label_id, tag):
        return MONAILabelClient(self.server_url, self.tmpdir, self.client_id).download_label(label_id, tag)

    def next_sample(self, strategy, params={}):
        return MONAILabelClient(self.server_url, self.tmpdir, self.client_id).next_sample(strategy, params)

    def create_session(self, image_in):
        return MONAILabelClient(self.server_url, self.tmpdir, self.client_id).create_session(image_in)

    def get_session(self, session_id):
        return MONAILabelClient(self.server_url, self.tmpdir, self.client_id).get_session(session_id)

    def remove_session(self, session_id):
        return MONAILabelClient(self.server_url, self.tmpdir, self.client_id).remove_session(session_id)

    def upload_image(self, image_in, image_id=None):
        return MONAILabelClient(self.server_url, self.tmpdir, self.client_id).upload_image(image_in, image_id)

    def save_label(self, image_in, label_in, params):
        return MONAILabelClient(self.server_url, self.tmpdir, self.client_id).save_label(
            image_in, label_in, params=params
        )

    def infer(self, model, image_in, params={}, label_in=None, file=None, session_id=None):
        logging.debug("Preparing input data for segmentation")
        self.reportProgress(0)

        client = MONAILabelClient(self.server_url, self.tmpdir, self.client_id)
        params["result_extension"] = ".nrrd"  # expect .nrrd
        params["result_dtype"] = "uint8"
        result_file, params = client.infer(model, image_in, params, label_in, file, session_id)

        logging.debug(f"Image Response: {result_file}")
        logging.debug(f"JSON  Response: {params}")

        self.reportProgress(100)
        return result_file, params

    def train_start(self, model=None, params={}):
        return MONAILabelClient(self.server_url, self.tmpdir, self.client_id).train_start(model, params)

    def train_status(self, check_if_running):
        return MONAILabelClient(self.server_url, self.tmpdir, self.client_id).train_status(check_if_running)

    def train_stop(self):
        return MONAILabelClient(self.server_url, self.tmpdir, self.client_id).train_stop()


class MONAILabelTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_MONAILabel1()

    def test_MONAILabel1(self):
        self.delayDisplay("Test passed")
