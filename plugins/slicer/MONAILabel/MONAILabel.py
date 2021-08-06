import copy
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
from collections import OrderedDict

import ctk
import qt
import SampleData
import SimpleITK as sitk
import sitkUtils
import slicer
import vtk
import vtkSegmentationCore
from MONAILabelLib import MONAILabelClient
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


class _ui_MONAILabelSettingsPanel(object):
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

        autoRunSegmentationCheckBox = qt.QCheckBox()
        autoRunSegmentationCheckBox.checked = True
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
        autoUpdateModelCheckBox.checked = True
        autoUpdateModelCheckBox.toolTip = "Enable this option to auto update model after submitting the label"
        groupLayout.addRow("Auto-Update Model:", autoUpdateModelCheckBox)
        parent.registerProperty(
            "MONAILabel/autoUpdateModel",
            ctk.ctkBooleanMapper(autoUpdateModelCheckBox, "checked", str(qt.SIGNAL("toggled(bool)"))),
            "valueAsInt",
            str(qt.SIGNAL("valueAsIntChanged(int)")),
        )

        vBoxLayout.addWidget(groupBox)
        vBoxLayout.addStretch(1)


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
        self._volumeNodes = []
        self._updatingGUIFromParameterNode = False
        self._scribblesEditorWidget = None

        self.info = {}
        self.models = OrderedDict()
        self.config = OrderedDict()
        self.current_sample = None
        self.samples = {}

        self.dgPositiveFiducialNode = None
        self.dgPositiveFiducialNodeObservers = []
        self.dgNegativeFiducialNode = None
        self.dgNegativeFiducialNodeObservers = []
        self.ignoreFiducialNodeAddEvent = False

        self.progressBar = None
        self.tmpdir = None
        self.timer = None

        self.scribblesMode = None

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

        self.ui.dgPositiveFiducialPlacementWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.dgPositiveFiducialPlacementWidget.placeButton().toolTip = "Select +ve points"
        self.ui.dgPositiveFiducialPlacementWidget.buttonsVisible = False
        self.ui.dgPositiveFiducialPlacementWidget.placeButton().show()
        self.ui.dgPositiveFiducialPlacementWidget.deleteButton().show()

        self.ui.dgNegativeFiducialPlacementWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.dgNegativeFiducialPlacementWidget.placeButton().toolTip = "Select -ve points"
        self.ui.dgNegativeFiducialPlacementWidget.buttonsVisible = False
        self.ui.dgNegativeFiducialPlacementWidget.placeButton().show()
        self.ui.dgNegativeFiducialPlacementWidget.deleteButton().show()

        # Connections
        self.ui.fetchServerInfoButton.connect("clicked(bool)", self.onClickFetchInfo)
        self.ui.serverComboBox.connect("currentIndexChanged(int)", self.onClickFetchInfo)
        self.ui.segmentationModelSelector.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.segmentationButton.connect("clicked(bool)", self.onClickSegmentation)
        self.ui.deepgrowModelSelector.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.nextSampleButton.connect("clicked(bool)", self.onNextSampleButton)
        self.ui.trainingButton.connect("clicked(bool)", self.onTraining)
        self.ui.stopTrainingButton.connect("clicked(bool)", self.onStopTraining)
        self.ui.trainingStatusButton.connect("clicked(bool)", self.onTrainingStatus)
        self.ui.saveLabelButton.connect("clicked(bool)", self.onSaveLabel)
        self.ui.uploadImageButton.connect("clicked(bool)", self.onUploadImage)
        self.ui.importLabelButton.connect("clicked(bool)", self.onImportLabel)

        # Scribbles
        # brush and eraser icon from: https://tablericons.com/
        self.ui.scribblesMethodSelector.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.paintScribblesButton.setIcon(self.icon("tool-brush.svg"))
        self.ui.paintScribblesButton.setToolTip("Paint scribbles for selected scribble layer")
        self.ui.eraseScribblesButton.setIcon(self.icon("eraser.svg"))
        self.ui.eraseScribblesButton.setToolTip("Erase scribbles for selected scribble layer")
        self.ui.updateScribblesButton.setIcon(self.icon("refresh-icon.png"))
        self.ui.updateScribblesButton.setToolTip(
            "Update label by sending scribbles to server to apply selected post processing method"
        )
        self.ui.selectForegroundButton.setIcon(self.icon("fg_green.svg"))
        self.ui.selectForegroundButton.setToolTip("Select foreground scribbles layer")
        self.ui.selectBackgroundButton.setIcon(self.icon("bg_red.svg"))
        self.ui.selectBackgroundButton.setToolTip("Select background scribbles layer")
        self.ui.selectedScribbleDisplay.setIcon(self.icon("gray.svg"))
        self.ui.selectedScribbleDisplay.setToolTip("Displaying selected scribbles layer")
        self.ui.selectedToolDisplay.setIcon(self.icon("gray.svg"))
        self.ui.selectedToolDisplay.setToolTip("Displaying selected scribbles tool")

        self.ui.brushSizeSlider.connect("valueChanged(double)", self.updateBrushSize)
        self.ui.brushSizeSlider.setToolTip("Change brush size for scribbles tool")
        self.ui.brush3dCheckbox.stateChanged.connect(self.on3dBrushCheckbox)
        self.ui.brush3dCheckbox.setToolTip("Use 3D brush to paint/erase in multiple slices in 3D")
        self.ui.updateScribblesButton.clicked.connect(self.onUpdateScribbles)
        self.ui.paintScribblesButton.clicked.connect(self.onPaintScribbles)
        self.ui.eraseScribblesButton.clicked.connect(self.onEraseScribbles)
        self.ui.selectForegroundButton.clicked.connect(self.onSelectForegroundScribbles)
        self.ui.selectBackgroundButton.clicked.connect(self.onSelectBackgroundScribbles)

        # start with scribbles section disabled
        self.ui.scribblesCollapsibleButton.setEnabled(False)
        self.ui.scribblesCollapsibleButton.setVisible(False)
        self.ui.scribblesCollapsibleButton.collapsed = True

        self.initializeParameterNode()
        self.updateServerUrlGUIFromSettings()
        # self.onClickFetchInfo()

    def cleanup(self):
        self.removeObservers()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def enter(self):
        self.initializeParameterNode()

    def exit(self):
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def scribblesLayersPresent(self):
        scribbles_exist = False
        if self._segmentNode is not None:
            segmentationNode = self._segmentNode
            segmentation = segmentationNode.GetSegmentation()
            numSegments = segmentation.GetNumberOfSegments()
            segmentIds = [segmentation.GetNthSegmentID(i) for i in range(numSegments)]
            scribbles_exist = sum([int("scribbles" in sid) for sid in segmentIds]) > 0
        return scribbles_exist

    def onStartScribbling(self):
        logging.debug("Scribbles start event")
        if (not self.scribblesLayersPresent()) and (self._scribblesEditorWidget is None):
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

            # create segmentEditorWidget to access "Paint" and "Erase" segmentation tools
            # these will be used to draw scribbles
            self._scribblesEditorWidget = slicer.qMRMLSegmentEditorWidget()
            self._scribblesEditorWidget.setMRMLScene(slicer.mrmlScene)
            segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()

            # adding new scribbles can overwrite a new one-hot vector, hence erase any existing
            # labels - this is not a desired behaviour hence we swith to overlay mode that enables drawing
            # scribbles without changing existing labels. Further explanation at:
            # https://discourse.slicer.org/t/how-can-i-set-masking-settings-on-a-segment-editor-effect-in-python/4406/7
            segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)

            # add all nodes to the widget
            slicer.mrmlScene.AddNode(segmentEditorNode)
            self._scribblesEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
            self._scribblesEditorWidget.setSegmentationNode(self._segmentNode)
            self._scribblesEditorWidget.setMasterVolumeNode(self._volumeNode)

    def onUpdateScribbles(self):
        logging.info("Scribbles update event")
        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

            # get scribbles + label
            segmentationNode = self._segmentNode
            labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
                segmentationNode, labelmapVolumeNode, self._volumeNode
            )
            scribbles_in = tempfile.NamedTemporaryFile(suffix=".nii.gz", dir=self.tmpdir).name
            self.reportProgress(5)

            # save scribbles + label to file
            slicer.util.saveNode(labelmapVolumeNode, scribbles_in)
            self.reportProgress(30)
            self.updateServerSettings()
            self.reportProgress(60)

            # send scribbles + label to server along with selected scribbles method
            scribblesMethod = self.ui.scribblesMethodSelector.currentText
            image_file = self.current_sample["id"]
            configs = self.getParamsFromConfig()
            result_file, params = self.logic.infer(scribblesMethod, image_file, configs.get("infer"), scribbles_in)

            # display result from server
            self.reportProgress(90)
            _, segment = self.currentSegment()
            label = segment.GetName()
            self.updateSegmentationMask(result_file, [label])
        except:
            slicer.util.errorDisplay(
                "Failed to post process label on MONAI Label Server using {}".format(scribblesMethod),
                detailedText=traceback.format_exc(),
            )
        finally:
            qt.QApplication.restoreOverrideCursor()
            self.reportProgress(100)

            # clear all temporary files
            if os.path.exists(scribbles_in):
                os.unlink(scribbles_in)

            if result_file and os.path.exists(result_file):
                os.unlink(result_file)

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
                logging.info("clearning {}".format(segmentId))
                labelMapRep = slicer.vtkOrientedImageData()
                segmentation.GetBinaryLabelmapRepresentation(segmentId, labelMapRep)
                vtkSegmentationCore.vtkOrientedImageDataResample.FillImage(labelMapRep, 0, labelMapRep.GetExtent())
                slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
                    labelMapRep, segmentation, segmentId, slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE
                )

    def onClearScribbles(self):
        # reset scribbles mode
        self.scribblesMode = None

        # clear scribbles editor widget
        if self._scribblesEditorWidget:
            widget = self._scribblesEditorWidget
            del widget
        self._scribblesEditorWidget = None

        # remove "scribbles" segments from label
        self.onClearScribblesSegmentNodes()

        # update tool/layer display
        self.updateScribblesStatusIcons()

        self.ui.scribblesCollapsibleButton.setEnabled(False)
        self.ui.scribblesCollapsibleButton.setVisible(False)
        self.ui.scribblesCollapsibleButton.collapsed = True

    def checkAndInitialiseScribbles(self):
        if self._scribblesEditorWidget is None:
            self.onStartScribbling()

        if self.scribblesMode is None:
            self.changeScribblesMode(tool="Paint", layer="foreground_scribbles")
            self.updateScribToolLayerFromMode()

    def updateScribToolLayerFromMode(self):
        logging.info("Scribbles mode {} ".format(self.scribblesMode))
        self.checkAndInitialiseScribbles()

        # update tool/layer select for scribblesEditorWidget
        tool, layer = self.getToolAndLayerFromScribblesMode()
        self._scribblesEditorWidget.setActiveEffectByName(tool)
        self._scribblesEditorWidget.setCurrentSegmentID(layer)

        # update brush type from checkbox
        is3dbrush = self.ui.brush3dCheckbox.checkState()
        self.on3dBrushCheckbox(state=is3dbrush)

        # update brush size from slider
        brushSize = self.ui.brushSizeSlider.value
        self.updateBrushSize(value=brushSize)

        # show user the selected tools
        self.updateScribblesStatusIcons()

    def getToolAndLayerFromScribblesMode(self):
        if self.scribblesMode is not None:
            return self.scribblesMode.split("+")
        else:
            # default modes
            return "Paint", "foreground_scribbles"

    def changeScribblesMode(self, tool=None, layer=None):
        ctool, clayer = self.getToolAndLayerFromScribblesMode()

        ctool = tool if tool != None else ctool
        clayer = layer if layer != None else clayer

        self.scribblesMode = "+".join([ctool, clayer])

    def onPaintScribbles(self):
        self.changeScribblesMode(tool="Paint")
        self.updateScribToolLayerFromMode()

    def onEraseScribbles(self):
        self.changeScribblesMode(tool="Erase")
        self.updateScribToolLayerFromMode()

    def onSelectForegroundScribbles(self):
        self.changeScribblesMode(layer="foreground_scribbles")
        self.updateScribToolLayerFromMode()

    def onSelectBackgroundScribbles(self):
        self.changeScribblesMode(layer="background_scribbles")
        self.updateScribToolLayerFromMode()

    def updateScribblesStatusIcons(self):
        if self.scribblesMode is None:
            self.ui.selectedScribbleDisplay.setIcon(self.icon("gray.svg"))
            self.ui.selectedToolDisplay.setIcon(self.icon("gray.svg"))
        else:
            tool, layer = self.getToolAndLayerFromScribblesMode()

            # update tool icon
            if tool == "Paint":
                self.ui.selectedToolDisplay.setIcon(self.icon("tool-brush.svg"))
            elif tool == "Erase":
                self.ui.selectedToolDisplay.setIcon(self.icon("eraser.svg"))

            # update scirbbles layer icon
            if layer == "foreground_scribbles":
                self.ui.selectedScribbleDisplay.setIcon(self.icon("fg_green.svg"))
            elif layer == "background_scribbles":
                self.ui.selectedScribbleDisplay.setIcon(self.icon("bg_red.svg"))

    def on3dBrushCheckbox(self, state):
        logging.info("3D brush update {}".format(state))
        self.checkAndInitialiseScribbles()
        effect = self._scribblesEditorWidget.activeEffect()

        # enable scribbles in 3d using a sphere brush
        effect.setParameter("BrushSphere", state)

    def updateBrushSize(self, value):
        logging.info("brush size update {}".format(value))
        self.checkAndInitialiseScribbles()
        effect = self._scribblesEditorWidget.activeEffect()

        # change scribbles brush size
        effect.setParameter("BrushAbsoluteDiameter", value)

    def onSceneStartClose(self, caller, event):
        self._volumeNode = None
        self._segmentNode = None
        self._volumeNodes.clear()
        self.setParameterNode(None)
        self.current_sample = None
        self.samples.clear()

        self.resetFiducial(
            self.ui.dgPositiveFiducialPlacementWidget, self.dgPositiveFiducialNode, self.dgPositiveFiducialNodeObservers
        )
        self.dgPositiveFiducialNode = None
        self.resetFiducial(
            self.ui.dgNegativeFiducialPlacementWidget, self.dgNegativeFiducialNode, self.dgNegativeFiducialNodeObservers
        )
        self.dgNegativeFiducialNode = None
        self.onClearScribbles()

    def resetFiducial(self, fiducialWidget, fiducialNode, fiducialNodeObservers):
        if fiducialWidget.placeModeEnabled:
            fiducialWidget.setPlaceModeEnabled(False)

        if fiducialNode:
            slicer.mrmlScene.RemoveNode(fiducialNode)
            self.removeFiducialNodeObservers(fiducialNode, fiducialNodeObservers)

    def onSceneEndClose(self, caller, event):
        if self.parent.isEntered:
            self.initializeParameterNode()

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

        # Update node selectors and sliders
        self.ui.inputSelector.clear()
        for v in self._volumeNodes:
            self.ui.inputSelector.addItem(v.GetName())
        if self._volumeNode:
            self.ui.inputSelector.setCurrentIndex(self.ui.inputSelector.findText(self._volumeNode.GetName()))
        self.ui.inputSelector.setEnabled(self._volumeNode is not None)

        self.updateSelector(self.ui.segmentationModelSelector, ["segmentation"], "SegmentationModel", 0)
        self.updateSelector(self.ui.deepgrowModelSelector, ["deepgrow"], "DeepgrowModel", 0)
        self.updateSelector(self.ui.scribblesMethodSelector, ["scribble"], "ScribbleMethod", 0)

        if self.models and [k for k, v in self.models.items() if v["type"] == "segmentation"]:
            self.ui.segmentationCollapsibleButton.collapsed = False
        if self.models and [k for k, v in self.models.items() if v["type"] == "deepgrow"]:
            self.ui.deepgrowCollapsibleButton.collapsed = False
        if self.models and [k for k, v in self.models.items() if v["type"] == "scribbles"]:
            self.ui.scribblesCollapsibleButton.collapsed = False

        self.ui.labelComboBox.clear()
        for label in self.info.get("labels", {}):
            self.ui.labelComboBox.addItem(label)
        currentLabel = self._parameterNode.GetParameter("CurrentLabel")
        idx = self.ui.labelComboBox.findText(currentLabel) if currentLabel else 0
        idx = 0 if idx < 0 < self.ui.labelComboBox.count else idx
        self.ui.labelComboBox.setCurrentIndex(idx)

        self.ui.appComboBox.clear()
        self.ui.appComboBox.addItem(self.info.get("name", ""))

        datastore_stats = self.info.get("datastore", {})
        current = datastore_stats.get("completed", 0)
        total = datastore_stats.get("total", 0)
        self.ui.activeLearningProgressBar.setValue(current / max(total, 1) * 100)
        self.ui.activeLearningProgressBar.setToolTip(f"{current}/{total} samples are labeled")

        train_stats = self.info.get("train_stats", {})
        dice = train_stats.get("best_metric", 0)
        self.updateAccuracyBar(dice)

        self.ui.strategyBox.clear()
        for strategy in self.info.get("strategies", {}):
            self.ui.strategyBox.addItem(strategy)
        currentStrategy = self._parameterNode.GetParameter("CurrentStrategy")
        self.ui.strategyBox.setCurrentIndex(self.ui.strategyBox.findText(currentStrategy) if currentStrategy else 0)

        # Show scribbles panel only if scribbles methods detected
        self.ui.scribblesCollapsibleButton.setVisible(self.ui.scribblesMethodSelector.count)

        # Enable/Disable
        self.ui.nextSampleButton.setEnabled(self.ui.strategyBox.count)

        is_training_running = True if self.info and self.isTrainingRunning() else False
        self.ui.trainingButton.setEnabled(self.info and not is_training_running)
        self.ui.stopTrainingButton.setEnabled(is_training_running)
        self.ui.trainingStatusButton.setEnabled(self.info)
        if is_training_running and self.timer is None:
            self.timer = qt.QTimer()
            self.timer.setInterval(5000)
            self.timer.connect("timeout()", self.monitorTraining)
            self.timer.start()

        self.ui.segmentationButton.setEnabled(
            self.ui.segmentationModelSelector.currentText and self._volumeNode is not None
        )
        self.ui.saveLabelButton.setEnabled(self._segmentNode is not None)
        self.ui.uploadImageButton.setEnabled(
            self.info and slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode") and self._segmentNode is None
        )
        self.ui.importLabelButton.setEnabled(self._segmentNode is not None)

        # Create empty markup fiducial node for deep grow +ve and -ve
        if self._segmentNode:
            if not self.dgPositiveFiducialNode:
                self.dgPositiveFiducialNode, self.dgPositiveFiducialNodeObservers = self.createFiducialNode(
                    "P", self.onDeepGrowFiducialNodeModified, [0.5, 1, 0.5]
                )
                self.ui.dgPositiveFiducialPlacementWidget.setCurrentNode(self.dgPositiveFiducialNode)
                self.ui.dgPositiveFiducialPlacementWidget.setPlaceModeEnabled(False)

            if not self.dgNegativeFiducialNode:
                self.dgNegativeFiducialNode, self.dgNegativeFiducialNodeObservers = self.createFiducialNode(
                    "N", self.onDeepGrowFiducialNodeModified, [0.5, 0.5, 1]
                )
                self.ui.dgNegativeFiducialPlacementWidget.setCurrentNode(self.dgNegativeFiducialNode)
                self.ui.dgNegativeFiducialPlacementWidget.setPlaceModeEnabled(False)

            self.ui.scribblesCollapsibleButton.setEnabled(self.ui.scribblesMethodSelector.count)
            self.ui.scribblesCollapsibleButton.collapsed = False

        self.ui.dgPositiveFiducialPlacementWidget.setEnabled(self.ui.deepgrowModelSelector.currentText)
        self.ui.dgNegativeFiducialPlacementWidget.setEnabled(self.ui.deepgrowModelSelector.currentText)

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

        currentStrategyIndex = self.ui.strategyBox.currentIndex
        if currentStrategyIndex >= 0:
            currentStrategy = self.ui.strategyBox.itemText(currentStrategyIndex)
            self._parameterNode.SetParameter("CurrentStrategy", currentStrategy)

        self._parameterNode.EndModify(wasModified)

    def updateSelector(self, selector, model_types, param, defaultIndex=0):
        wasSelectorBlocked = selector.blockSignals(True)
        selector.clear()

        for model_name, model in self.models.items():
            if model["type"] in model_types:
                selector.addItem(model_name)
                selector.setItemData(selector.count - 1, model["description"], qt.Qt.ToolTipRole)

        model = self._parameterNode.GetParameter(param)
        model = "" if not model else model
        modelIndex = selector.findText(model)
        modelIndex = defaultIndex if modelIndex < 0 < selector.count else modelIndex
        selector.setCurrentIndex(modelIndex)

        try:
            modelInfo = self.models[model]
            selector.setToolTip(modelInfo["description"])
        except:
            selector.setToolTip("")
        selector.blockSignals(wasSelectorBlocked)

    def updateConfigTable(self):
        table = self.ui.configTable
        table.clear()
        table.setHorizontalHeaderLabels(["section", "name", "value"])

        config = copy.deepcopy(self.config)
        infer = config.get("infer", {})
        train = config.get("train", {})
        activelearning = config.get("activelearning", {})
        table.setRowCount(len(infer) + len(activelearning) + len(train))

        config = {"infer": infer, "activelearning": activelearning, "train": train}
        colors = {
            "infer": qt.QColor(255, 255, 255),
            "activelearning": qt.QColor(220, 220, 220),
            "train": qt.QColor(255, 255, 255),
        }

        n = 0
        for section in config:
            table.setSpan(n, 0, n + len(config[section]), 1)
            for key in config[section]:
                table.setItem(n, 0, qt.QTableWidgetItem(section))
                table.setItem(n, 1, qt.QTableWidgetItem(key))

                val = config[section][key]
                if isinstance(val, dict) or isinstance(val, list):
                    combo = qt.QComboBox()
                    for m, v in enumerate(val):
                        combo.addItem(v)
                    combo.setCurrentIndex(0)
                    table.setCellWidget(n, 2, combo)
                elif isinstance(val, bool):
                    checkbox = qt.QCheckBox()
                    checkbox.setChecked(val)
                    table.setCellWidget(n, 2, checkbox)
                else:
                    table.setItem(n, 2, qt.QTableWidgetItem(str(val)))

                table.item(n, 0).setBackground(colors[section])
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

    def getParamsFromConfig(self):
        config = {}
        for row in range(self.ui.configTable.rowCount):
            section = str(self.ui.configTable.item(row, 0).text())
            name = str(self.ui.configTable.item(row, 1).text())
            value = self.ui.configTable.item(row, 2)
            if value is None:
                value = self.ui.configTable.cellWidget(row, 2)
                value = value.checked if isinstance(value, qt.QCheckBox) else value.currentText
            else:
                value = str(value.text())

            v = self.config.get(section, {}).get(name, {})
            if isinstance(v, int):
                value = int(value)
            elif isinstance(v, float):
                value = float(value)

            if config.get(section) is None:
                config[section] = {}
            config[section][name] = value
            # print(f"row: {row}, section: {section}, name: {name}, value: {value}, type: {type(v)}")
        return config

    def onDeepGrowFiducialNodeModified(self, observer, eventid):
        logging.debug("Deepgrow Point Event!!")

        if self.ignoreFiducialNodeAddEvent:
            return

        markupsNode = observer
        movingMarkupIndex = markupsNode.GetDisplayNode().GetActiveControlPoint()
        logging.debug("Markup point added; point ID = {}".format(movingMarkupIndex))

        current_point = self.getFiducialPointXYZ(markupsNode, movingMarkupIndex)
        self.onClickDeepgrow(current_point)

        self.ignoreFiducialNodeAddEvent = True
        self.onEditFiducialPoints(self.dgPositiveFiducialNode, "MONAILabel.ForegroundPoints")
        self.onEditFiducialPoints(self.dgNegativeFiducialNode, "MONAILabel.BackgroundPoints")
        self.ignoreFiducialNodeAddEvent = False

    def getFiducialPointsXYZ(self, fiducialNode):
        v = self._volumeNode
        RasToIjkMatrix = vtk.vtkMatrix4x4()
        v.GetRASToIJKMatrix(RasToIjkMatrix)

        point_set = []
        n = fiducialNode.GetNumberOfFiducials()
        for i in range(n):
            coord = [0.0, 0.0, 0.0]
            fiducialNode.GetNthFiducialPosition(i, coord)

            world = [0, 0, 0, 0]
            fiducialNode.GetNthFiducialWorldCoordinates(i, world)

            p_Ras = [coord[0], coord[1], coord[2], 1.0]
            p_Ijk = RasToIjkMatrix.MultiplyDoublePoint(p_Ras)
            p_Ijk = [round(i) for i in p_Ijk]

            logging.debug("RAS: {}; WORLD: {}; IJK: ".format(coord, world, p_Ijk))
            point_set.append(p_Ijk[0:3])

        logging.info("Current Fiducials-Points: {}".format(point_set))
        return point_set

    def getFiducialPointXYZ(self, fiducialNode, index):
        v = self._volumeNode
        RasToIjkMatrix = vtk.vtkMatrix4x4()
        v.GetRASToIJKMatrix(RasToIjkMatrix)

        coord = [0.0, 0.0, 0.0]
        fiducialNode.GetNthFiducialPosition(index, coord)

        world = [0, 0, 0, 0]
        fiducialNode.GetNthFiducialWorldCoordinates(index, world)

        p_Ras = [coord[0], coord[1], coord[2], 1.0]
        p_Ijk = RasToIjkMatrix.MultiplyDoublePoint(p_Ras)
        p_Ijk = [round(i) for i in p_Ijk]

        logging.debug("RAS: {}; WORLD: {}; IJK: ".format(coord, world, p_Ijk))
        return p_Ijk[0:3]

    def onEditFiducialPoints(self, fiducialNode, tagName):
        if fiducialNode is None:
            return

        fiducialNode.RemoveAllMarkups()
        segmentId, segment = self.currentSegment()
        if segment and segmentId:
            v = self._volumeNode
            IjkToRasMatrix = vtk.vtkMatrix4x4()
            v.GetIJKToRASMatrix(IjkToRasMatrix)

            fPosStr = vtk.mutable("")
            segment.GetTag(tagName, fPosStr)
            pointset = str(fPosStr)
            logging.debug("{} => {} Fiducial points are: {}".format(segmentId, segment.GetName(), pointset))

            if fPosStr is not None and len(pointset) > 0:
                points = json.loads(pointset)
                for p in points:
                    p_Ijk = [p[0], p[1], p[2], 1.0]
                    p_Ras = IjkToRasMatrix.MultiplyDoublePoint(p_Ijk)
                    logging.debug("Add Fiducial: {} => {}".format(p_Ijk, p_Ras))
                    fiducialNode.AddFiducialFromArray(p_Ras[0:3])

    def currentSegment(self):
        segmentation = self._segmentNode.GetSegmentation()
        segmentId = segmentation.GetSegmentIdBySegmentName(self.ui.labelComboBox.currentText)
        segment = segmentation.GetSegment(segmentId)

        logging.debug("Current SegmentID: {}; Segment: {}".format(segmentId, segment))
        return segmentId, segment

    def icon(self, name="MONAILabel.png"):
        # It should not be necessary to modify this method
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", name)
        if os.path.exists(iconPath):
            return qt.QIcon(iconPath)
        return qt.QIcon()

    def updateServerSettings(self):
        self.logic.setServer(self.serverUrl())
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

        # if self._volumeNode is None:
        #    self.onNextSampleButton()

    def fetchInfo(self, showInfo=False):
        if not self.logic:
            return

        start = time.time()
        try:
            self.updateServerSettings()
            info = self.logic.info()
            self.info = info

            models = info["models"]
        except:
            slicer.util.errorDisplay(
                "Failed to fetch models from remote server. "
                "Make sure server address is correct and <server_uri>/info/ "
                "is accessible in browser",
                detailedText=traceback.format_exc(),
            )
            return

        self.models.clear()
        self.config = info["config"]
        model_count = {}
        for k, v in models.items():
            model_type = v.get("type", "segmentation")
            model_count[model_type] = model_count.get(model_type, 0) + 1

            logging.debug("{} = {}".format(k, model_type))
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
        logging.debug(msg)
        logging.info("Time consumed by fetch info: {0:3.1f}".format(time.time() - start))

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
            configs = self.getParamsFromConfig()
            status = self.logic.train_start(configs.get("train"))

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

        logging.info("Time consumed by training: {0:3.1f}".format(time.time() - start))

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

        logging.info("Time consumed by stop training: {0:3.1f}".format(time.time() - start))

    def isTrainingRunning(self, check_only=True):
        if not self.logic:
            return False
        self.updateServerSettings()
        return self.logic.train_status(check_only)

    def onTrainingStatus(self):
        if not self.logic:
            return

        start = time.time()
        status = None
        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            self.updateServerSettings()
            status = self.logic.train_status(False)
        except:
            logging.debug("Failed to fetch Training Status\n{}".format(traceback.format_exc()))
        finally:
            qt.QApplication.restoreOverrideCursor()

        if status:
            result = status.get("details", [])[-1]
            try:
                result = json.loads(result)
                result = json.dumps(result, indent=2)
            except:
                pass
            msg = "Status: {}\nStart Time: {}\nEnd Time: {}\nResult: {}".format(
                status.get("status"), status.get("start_ts"), status.get("end_ts"), status.get("result", result)
            )
            slicer.util.infoDisplay(msg, detailedText=json.dumps(status, indent=2))
        else:
            slicer.util.errorDisplay("No Training Tasks Found\t")

        self.updateGUIFromParameterNode()
        logging.info("Time consumed by next_sample: {0:3.1f}".format(time.time() - start))

    def onNextSampleButton(self):
        if not self.logic:
            return

        if self._volumeNode or len(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")):
            if not slicer.util.confirmOkCancelDisplay(
                "This will close current scene.  Please make sure you have saved your current work.\n"
                "Are you sure to continue?"
            ):
                return
            self.onClearScribbles()
            slicer.mrmlScene.Clear(0)

        start = time.time()
        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

            self.updateServerSettings()
            configs = self.getParamsFromConfig()
            strategy = self.ui.strategyBox.currentText
            if not strategy:
                slicer.util.errorDisplay("No Strategy Found/Selected\t")
                return

            sample = self.logic.next_sample(strategy, configs.get("activelearning"))
            logging.debug(sample)

            if self.samples.get(sample["id"]) is not None:
                self.current_sample = self.samples[sample["id"]]
                name = self.current_sample["VolumeNodeName"]
                index = self.ui.inputSelector.findText(name)
                self.ui.inputSelector.setCurrentIndex(index)
                return

            image_file = sample["path"].replace("/workspace", "/raid/sachi")
            logging.info(f"Check if file exists/shared locally: {image_file}")
            if os.path.exists(image_file):
                self._volumeNode = slicer.util.loadVolume(image_file)
            else:
                download_uri = f"{self.serverUrl()}{sample['url']}"
                logging.info(download_uri)

                sampleDataLogic = SampleData.SampleDataLogic()
                self._volumeNode = sampleDataLogic.downloadFromURL(
                    nodeNames=sample["name"], fileNames=sample["name"], uris=download_uri, checksums=sample["checksum"]
                )[0]

            self.initSample(sample)

        except:
            slicer.util.errorDisplay(
                "Failed to fetch Sample from MONAI Label Server", detailedText=traceback.format_exc()
            )
        finally:
            qt.QApplication.restoreOverrideCursor()

        self.updateGUIFromParameterNode()
        logging.info("Time consumed by next_sample: {0:3.1f}".format(time.time() - start))

    def initSample(self, sample, autosegment=True):
        sample["VolumeNodeName"] = self._volumeNode.GetName()
        self.current_sample = sample
        self.samples[sample["id"]] = sample
        self._volumeNodes.append(self._volumeNode)

        # Create Empty Segments for all labels for this node
        self.createSegmentNode()
        segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
        segmentEditorWidget.setSegmentationNode(self._segmentNode)
        segmentEditorWidget.setMasterVolumeNode(self._volumeNode)

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

    def onUploadImage(self):
        volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        name = volumeNode.GetName()
        image_id = f"{name}.nii.gz"
        if not slicer.util.confirmOkCancelDisplay(
            f"This will push/update volume to MONAILabel server as {image_id}\n" "Are you sure to continue?"
        ):
            return

        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            in_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", dir=self.tmpdir).name
            self.reportProgress(5)

            start = time.time()
            slicer.util.saveNode(volumeNode, in_file)
            logging.info("Saved Input Node into {0} in {1:3.1f}s".format(in_file, time.time() - start))
            self.reportProgress(30)

            self.logic.upload_image(in_file, image_id)
            self.reportProgress(100)

            self._volumeNode = volumeNode
            self.initSample({"id": image_id}, autosegment=False)
            qt.QApplication.restoreOverrideCursor()

            self.updateGUIFromParameterNode()
        except:
            self.reportProgress(100)
            qt.QApplication.restoreOverrideCursor()
            slicer.util.errorDisplay("Failed to upload volume to Server", detailedText=traceback.format_exc())

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
        self.onClearScribbles()
        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            segmentationNode = self._segmentNode
            labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
                segmentationNode, labelmapVolumeNode, self._volumeNode
            )

            label_in = tempfile.NamedTemporaryFile(suffix=".nii.gz", dir=self.tmpdir).name
            self.reportProgress(5)

            slicer.util.saveNode(labelmapVolumeNode, label_in)
            self.reportProgress(30)

            self.updateServerSettings()
            result = self.logic.save_label(self.current_sample["id"], label_in)
            self.fetchInfo()

            if slicer.util.settingsValue("MONAILabel/autoUpdateModel", True, converter=slicer.util.toBool):
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

        logging.info("Time consumed by save label: {0:3.1f}".format(time.time() - start))

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

            configs = self.getParamsFromConfig()
            result_file, params = self.logic.infer(model, image_file, configs.get("infer"))

            self.updateSegmentationMask(result_file, self.models[model].get("labels"))
        except:
            slicer.util.errorDisplay(
                "Failed to run inference in MONAI Label Server", detailedText=traceback.format_exc()
            )
        finally:
            qt.QApplication.restoreOverrideCursor()
            if result_file and os.path.exists(result_file):
                os.unlink(result_file)

        self.updateGUIFromParameterNode()
        logging.info("Time consumed by segmentation: {0:3.1f}".format(time.time() - start))

    def onClickDeepgrow(self, current_point):
        model = self.ui.deepgrowModelSelector.currentText
        if not model:
            slicer.util.warningDisplay("Please select a deepgrow model")
            return

        _, segment = self.currentSegment()
        if not segment:
            slicer.util.warningDisplay("Please add the required label to run deepgrow")
            return

        foreground_all = self.getFiducialPointsXYZ(self.dgPositiveFiducialNode)
        background_all = self.getFiducialPointsXYZ(self.dgNegativeFiducialNode)

        segment.SetTag("MONAILabel.ForegroundPoints", json.dumps(foreground_all))
        segment.SetTag("MONAILabel.BackgroundPoints", json.dumps(background_all))

        # use model info "deepgrow" to determine
        deepgrow_3d = False if self.models[model].get("dimension", 3) == 2 else True
        start = time.time()

        label = segment.GetName()
        operationDescription = "Run Deepgrow for segment: {}; model: {}; 3d {}".format(label, model, deepgrow_3d)
        logging.debug(operationDescription)

        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

            sliceIndex = current_point[2]
            logging.debug("Slice Index: {}".format(sliceIndex))

            if deepgrow_3d:
                foreground = foreground_all
                background = background_all
            else:
                foreground = [x for x in foreground_all if x[2] == sliceIndex]
                background = [x for x in background_all if x[2] == sliceIndex]

            logging.debug("Foreground: {}".format(foreground))
            logging.debug("Background: {}".format(background))
            logging.debug("Current point: {}".format(current_point))

            image_file = self.current_sample["id"]
            params = {
                "foreground": foreground,
                "background": background,
            }

            configs = self.getParamsFromConfig()
            params.update(configs.get("infer", {}))
            result_file, params = self.logic.infer(model, image_file, params)
            logging.debug("Params from deepgrow is {}".format(params))

            self.updateSegmentationMask(result_file, [label], None if deepgrow_3d else sliceIndex)
        except:
            logging.exception("Unknown Exception")
            slicer.util.errorDisplay(operationDescription + " - unexpected error.", detailedText=traceback.format_exc())
        finally:
            qt.QApplication.restoreOverrideCursor()

        self.updateGUIFromParameterNode()
        logging.info("Time consumed by Deepgrow: {0:3.1f}".format(time.time() - start))

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

    def getLabelColor(self, name):
        color = GenericAnatomyColors.get(name.lower())
        return [c / 255.0 for c in color] if color else None

    def updateSegmentationMask(self, in_file, labels, sliceIndex=None):
        # TODO:: Add ROI Node (for Bounding Box if provided in the result)

        start = time.time()
        logging.debug("Update Segmentation Mask from: {}".format(in_file))
        if in_file and not os.path.exists(in_file):
            return False

        segmentationNode = self._segmentNode
        segmentation = segmentationNode.GetSegmentation()

        if in_file is None:
            for label in labels:
                if not segmentation.GetSegmentIdBySegmentName(label):
                    segmentation.AddEmptySegment(label, label, self.getLabelColor(label))
            return True

        # segmentId, segment = self.currentSegment()
        labelImage = sitk.ReadImage(in_file)
        labelmapVolumeNode = sitkUtils.PushVolumeToSlicer(labelImage, None, className="vtkMRMLLabelMapVolumeNode")

        existing_label_ids = {}
        for label in labels:
            id = segmentation.GetSegmentIdBySegmentName(label)
            if id:
                existing_label_ids[label] = id

        numberOfExistingSegments = segmentation.GetNumberOfSegments()
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segmentationNode)
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

        numberOfAddedSegments = segmentation.GetNumberOfSegments() - numberOfExistingSegments
        logging.debug("Adding {} segments".format(numberOfAddedSegments))

        addedSegmentIds = [
            segmentation.GetNthSegmentID(numberOfExistingSegments + i) for i in range(numberOfAddedSegments)
        ]
        for i, segmentId in enumerate(addedSegmentIds):
            segment = segmentation.GetSegment(segmentId)
            print("Setting new segmentation with id: {} => {}".format(segmentId, segment.GetName()))

            label = labels[i] if i < len(labels) else "unknown {}".format(i)
            # segment.SetName(label)
            # segment.SetColor(self.getLabelColor(label))

            if label in existing_label_ids:
                segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
                segmentEditorWidget.setSegmentationNode(segmentationNode)
                segmentEditorWidget.setMasterVolumeNode(self._volumeNode)
                segmentEditorWidget.setCurrentSegmentID(existing_label_ids[label])

                effect = segmentEditorWidget.effectByName("Logical operators")
                labelmap = slicer.vtkOrientedImageData()
                segmentationNode.GetBinaryLabelmapRepresentation(segmentId, labelmap)

                if sliceIndex:
                    selectedSegmentLabelmap = effect.selectedSegmentLabelmap()
                    dims = selectedSegmentLabelmap.GetDimensions()
                    count = 0
                    for x in range(dims[0]):
                        for y in range(dims[1]):
                            if selectedSegmentLabelmap.GetScalarComponentAsDouble(x, y, sliceIndex, 0):
                                count = count + 1
                            selectedSegmentLabelmap.SetScalarComponentFromDouble(x, y, sliceIndex, 0, 0)

                    logging.debug("Total Non Zero: {}".format(count))

                    # Clear the Slice
                    if count:
                        effect.modifySelectedSegmentByLabelmap(
                            selectedSegmentLabelmap, slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeSet
                        )

                    # Union label map
                    effect.modifySelectedSegmentByLabelmap(
                        labelmap, slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeAdd
                    )
                else:
                    # adding bypass masking to not overwrite other layers,
                    # needed for preserving scribbles during updates
                    # help from: https://github.com/Slicer/Slicer/blob/master/Modules/Loadable/Segmentations/EditorEffects/Python/SegmentEditorLogicalEffect.py
                    bypassMask = True
                    effect.modifySelectedSegmentByLabelmap(
                        labelmap, slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeSet, bypassMask
                    )

                segmentationNode.RemoveSegment(segmentId)

        self.showSegmentationsIn3D()
        logging.info("Time consumed by updateSegmentationMask: {0:3.1f}".format(time.time() - start))
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

    def createFiducialNode(self, name, onMarkupNodeModified, color):
        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsDisplayNode")
        displayNode.SetTextScale(0)
        displayNode.SetSelectedColor(color)

        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        fiducialNode.SetName(name)
        fiducialNode.SetAndObserveDisplayNodeID(displayNode.GetID())

        fiducialNodeObservers = []
        self.addFiducialNodeObserver(fiducialNode, onMarkupNodeModified)
        return fiducialNode, fiducialNodeObservers

    def removeFiducialNodeObservers(self, fiducialNode, fiducialNodeObservers):
        if fiducialNode and fiducialNodeObservers:
            for observer in fiducialNodeObservers:
                fiducialNode.RemoveObserver(observer)

    def addFiducialNodeObserver(self, fiducialNode, onMarkupNodeModified):
        fiducialNodeObservers = []
        if fiducialNode:
            eventIds = [slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent]
            for eventId in eventIds:
                fiducialNodeObservers.append(fiducialNode.AddObserver(eventId, onMarkupNodeModified))
        return fiducialNodeObservers


class MONAILabelLogic(ScriptedLoadableModuleLogic):
    def __init__(self, tmpdir=None, server_url=None, progress_callback=None):
        ScriptedLoadableModuleLogic.__init__(self)

        self.tmpdir = slicer.util.tempDirectory("slicer-monai-label") if tmpdir is None else tmpdir
        self.volumeToSessions = dict()
        self.progress_callback = progress_callback

        self.server_url = server_url
        self.useCompression = True
        self.useSession = False

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
        if not server_url:
            server_url = "http://127.0.0.1:8000"
        self.server_url = server_url

    def setProgressCallback(self, progress_callback=None):
        self.progress_callback = progress_callback

    def reportProgress(self, progress):
        if self.progress_callback:
            self.progress_callback(progress)

    def info(self):
        return MONAILabelClient(self.server_url, self.tmpdir).info()

    def next_sample(self, strategy, params={}):
        return MONAILabelClient(self.server_url, self.tmpdir).next_sample(strategy, params)

    def upload_image(self, image_in, image_id=None):
        return MONAILabelClient(self.server_url, self.tmpdir).upload_image(image_in, image_id)

    def save_label(self, image_in, label_in):
        return MONAILabelClient(self.server_url, self.tmpdir).save_label(image_in, label_in)

    def infer(self, model, image_in, params={}, label_in=None):
        logging.debug("Preparing input data for segmentation")
        self.reportProgress(0)

        client = MONAILabelClient(self.server_url, self.tmpdir)
        result_file, params = client.infer(model, image_in, params, label_in)

        logging.debug(f"Image Response: {result_file}")
        logging.debug(f"JSON  Response: {params}")

        self.reportProgress(100)
        return result_file, params

    def train_start(self, params={}):
        return MONAILabelClient(self.server_url, self.tmpdir).train_start(params)

    def train_status(self, check_if_running):
        return MONAILabelClient(self.server_url, self.tmpdir).train_status(check_if_running)

    def train_stop(self):
        return MONAILabelClient(self.server_url, self.tmpdir).train_stop()


class MONAILabelTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_MONAILabel1()

    def test_MONAILabel1(self):
        self.delayDisplay("Test passed")


GenericAnatomyColors = {
    "background": (0, 0, 0),
    "tissue": (128, 174, 128),
    "bone": (241, 214, 145),
    "skin": (177, 122, 101),
    "connective tissue": (111, 184, 210),
    "blood": (216, 101, 79),
    "organ": (221, 130, 101),
    "mass": (144, 238, 144),
    "muscle": (192, 104, 88),
    "foreign object": (220, 245, 20),
    "waste": (78, 63, 0),
    "teeth": (255, 250, 220),
    "fat": (230, 220, 70),
    "gray matter": (200, 200, 235),
    "white matter": (250, 250, 210),
    "nerve": (244, 214, 49),
    "vein": (0, 151, 206),
    "artery": (216, 101, 79),
    "capillary": (183, 156, 220),
    "ligament": (183, 214, 211),
    "tendon": (152, 189, 207),
    "cartilage": (111, 184, 210),
    "meniscus": (178, 212, 242),
    "lymph node": (68, 172, 100),
    "lymphatic vessel": (111, 197, 131),
    "cerebro-spinal fluid": (85, 188, 255),
    "bile": (0, 145, 30),
    "urine": (214, 230, 130),
    "feces": (78, 63, 0),
    "gas": (218, 255, 255),
    "fluid": (170, 250, 250),
    "edema": (140, 224, 228),
    "bleeding": (188, 65, 28),
    "necrosis": (216, 191, 216),
    "clot": (145, 60, 66),
    "embolism": (150, 98, 83),
    "head": (177, 122, 101),
    "central nervous system": (244, 214, 49),
    "brain": (250, 250, 225),
    "gray matter of brain": (200, 200, 215),
    "telencephalon": (68, 131, 98),
    "cerebral cortex": (128, 174, 128),
    "right frontal lobe": (83, 146, 164),
    "left frontal lobe": (83, 146, 164),
    "right temporal lobe": (162, 115, 105),
    "left temporal lobe": (162, 115, 105),
    "right parietal lobe": (141, 93, 137),
    "left parietal lobe": (141, 93, 137),
    "right occipital lobe": (182, 166, 110),
    "left occipital lobe": (182, 166, 110),
    "right insular lobe": (188, 135, 166),
    "left insular lobe": (188, 135, 166),
    "right limbic lobe": (154, 150, 201),
    "left limbic lobe": (154, 150, 201),
    "right striatum": (177, 140, 190),
    "left striatum": (177, 140, 190),
    "right caudate nucleus": (30, 111, 85),
    "left caudate nucleus": (30, 111, 85),
    "right putamen": (210, 157, 166),
    "left putamen": (210, 157, 166),
    "right pallidum": (48, 129, 126),
    "left pallidum": (48, 129, 126),
    "right amygdaloid complex": (98, 153, 112),
    "left amygdaloid complex": (98, 153, 112),
    "diencephalon": (69, 110, 53),
    "thalamus": (166, 113, 137),
    "right thalamus": (122, 101, 38),
    "left thalamus": (122, 101, 38),
    "pineal gland": (253, 135, 192),
    "midbrain": (145, 92, 109),
    "substantia nigra": (46, 101, 131),
    "right substantia nigra": (0, 108, 112),
    "left substantia nigra": (0, 108, 112),
    "cerebral white matter": (250, 250, 225),
    "right superior longitudinal fasciculus": (127, 150, 88),
    "left superior longitudinal fasciculus": (127, 150, 88),
    "right inferior longitudinal fasciculus": (159, 116, 163),
    "left inferior longitudinal fasciculus": (159, 116, 163),
    "right arcuate fasciculus": (125, 102, 154),
    "left arcuate fasciculus": (125, 102, 154),
    "right uncinate fasciculus": (106, 174, 155),
    "left uncinate fasciculus": (106, 174, 155),
    "right cingulum bundle": (154, 146, 83),
    "left cingulum bundle": (154, 146, 83),
    "projection fibers": (126, 126, 55),
    "right corticospinal tract": (201, 160, 133),
    "left corticospinal tract": (201, 160, 133),
    "right optic radiation": (78, 152, 141),
    "left optic radiation": (78, 152, 141),
    "right medial lemniscus": (174, 140, 103),
    "left medial lemniscus": (174, 140, 103),
    "right superior cerebellar peduncle": (139, 126, 177),
    "left superior cerebellar peduncle": (139, 126, 177),
    "right middle cerebellar peduncle": (148, 120, 72),
    "left middle cerebellar peduncle": (148, 120, 72),
    "right inferior cerebellar peduncle": (186, 135, 135),
    "left inferior cerebellar peduncle": (186, 135, 135),
    "optic chiasm": (99, 106, 24),
    "right optic tract": (156, 171, 108),
    "left optic tract": (156, 171, 108),
    "right fornix": (64, 123, 147),
    "left fornix": (64, 123, 147),
    "commissural fibers": (138, 95, 74),
    "corpus callosum": (97, 113, 158),
    "posterior commissure": (126, 161, 197),
    "cerebellar white matter": (194, 195, 164),
    "CSF space": (85, 188, 255),
    "ventricles of brain": (88, 106, 215),
    "right lateral ventricle": (88, 106, 215),
    "left lateral ventricle": (88, 106, 215),
    "right third ventricle": (88, 106, 215),
    "left third ventricle": (88, 106, 215),
    "cerebral aqueduct": (88, 106, 215),
    "fourth ventricle": (88, 106, 215),
    "subarachnoid space": (88, 106, 215),
    "spinal cord": (244, 214, 49),
    "gray matter of spinal cord": (200, 200, 215),
    "white matter of spinal cord": (250, 250, 225),
    "endocrine system of brain": (82, 174, 128),
    "pituitary gland": (57, 157, 110),
    "adenohypophysis": (60, 143, 83),
    "neurohypophysis": (92, 162, 109),
    "meninges": (255, 244, 209),
    "dura mater": (255, 244, 209),
    "arachnoid": (255, 244, 209),
    "pia mater": (255, 244, 209),
    "muscles of head": (201, 121, 77),
    "salivary glands": (70, 163, 117),
    "lips": (188, 91, 95),
    "nose": (177, 122, 101),
    "tongue": (166, 84, 94),
    "soft palate": (182, 105, 107),
    "right inner ear": (229, 147, 118),
    "left inner ear": (229, 147, 118),
    "right external ear": (174, 122, 90),
    "left external ear": (174, 122, 90),
    "right middle ear": (201, 112, 73),
    "left middle ear": (201, 112, 73),
    "right eyeball": (194, 142, 0),
    "left eyeball": (194, 142, 0),
    "skull": (241, 213, 144),
    "right frontal bone": (203, 179, 77),
    "left frontal bone": (203, 179, 77),
    "right parietal bone": (229, 204, 109),
    "left parietal bone": (229, 204, 109),
    "right temporal bone": (255, 243, 152),
    "left temporal bone": (255, 243, 152),
    "right sphenoid bone": (209, 185, 85),
    "left sphenoid bone": (209, 185, 85),
    "right ethmoid bone": (248, 223, 131),
    "left ethmoid bone": (248, 223, 131),
    "occipital bone": (255, 230, 138),
    "maxilla": (196, 172, 68),
    "right zygomatic bone": (255, 255, 167),
    "right lacrimal bone": (255, 250, 160),
    "vomer bone": (255, 237, 145),
    "right palatine bone": (242, 217, 123),
    "left palatine bone": (242, 217, 123),
    "mandible": (222, 198, 101),
    "neck": (177, 122, 101),
    "muscles of neck": (213, 124, 109),
    "pharynx": (184, 105, 108),
    "larynx": (150, 208, 243),
    "thyroid gland": (62, 162, 114),
    "right parathyroid glands": (62, 162, 114),
    "left parathyroid glands": (62, 162, 114),
    "skeleton of neck": (242, 206, 142),
    "hyoid bone": (250, 210, 139),
    "cervical vertebral column": (255, 255, 207),
    "thorax": (177, 122, 101),
    "trachea": (182, 228, 255),
    "bronchi": (175, 216, 244),
    "right lung": (197, 165, 145),
    "left lung": (197, 165, 145),
    "superior lobe of right lung": (172, 138, 115),
    "superior lobe of left lung": (172, 138, 115),
    "middle lobe of right lung": (202, 164, 140),
    "inferior lobe of right lung": (224, 186, 162),
    "inferior lobe of left lung": (224, 186, 162),
    "pleura": (255, 245, 217),
    "heart": (206, 110, 84),
    "right atrium": (210, 115, 89),
    "left atrium": (203, 108, 81),
    "atrial septum": (233, 138, 112),
    "ventricular septum": (195, 100, 73),
    "right ventricle of heart": (181, 85, 57),
    "left ventricle of heart": (152, 55, 13),
    "mitral valve": (159, 63, 27),
    "tricuspid valve": (166, 70, 38),
    "aortic valve": (218, 123, 97),
    "pulmonary valve": (225, 130, 104),
    "aorta": (224, 97, 76),
    "pericardium": (255, 244, 209),
    "pericardial cavity": (184, 122, 154),
    "esophagus": (211, 171, 143),
    "thymus": (47, 150, 103),
    "mediastinum": (255, 244, 209),
    "skin of thoracic wall": (173, 121, 88),
    "muscles of thoracic wall": (188, 95, 76),
    "skeleton of thorax": (255, 239, 172),
    "thoracic vertebral column": (226, 202, 134),
    "ribs": (253, 232, 158),
    "sternum": (244, 217, 154),
    "right clavicle": (205, 179, 108),
    "left clavicle": (205, 179, 108),
    "abdominal cavity": (186, 124, 161),
    "abdomen": (177, 122, 101),
    "peritoneum": (255, 255, 220),
    "omentum": (234, 234, 194),
    "peritoneal cavity": (204, 142, 178),
    "retroperitoneal space": (180, 119, 153),
    "stomach": (216, 132, 105),
    "duodenum": (255, 253, 229),
    "small bowel": (205, 167, 142),
    "colon": (204, 168, 143),
    "anus": (255, 224, 199),
    "liver": (221, 130, 101),
    "biliary tree": (0, 145, 30),
    "gallbladder": (139, 150, 98),
    "pancreas": (249, 180, 111),
    "spleen": (157, 108, 162),
    "urinary system": (203, 136, 116),
    "right kidney": (185, 102, 83),
    "left kidney": (185, 102, 83),
    "right ureter": (247, 182, 164),
    "left ureter": (247, 182, 164),
    "urinary bladder": (222, 154, 132),
    "urethra": (124, 186, 223),
    "right adrenal gland": (249, 186, 150),
    "left adrenal gland": (249, 186, 150),
    "female internal genitalia": (244, 170, 147),
    "uterus": (255, 181, 158),
    "right fallopian tube": (255, 190, 165),
    "left fallopian tube": (227, 153, 130),
    "right ovary": (213, 141, 113),
    "left ovary": (213, 141, 113),
    "vagina": (193, 123, 103),
    "male internal genitalia": (216, 146, 127),
    "prostate": (230, 158, 140),
    "right seminal vesicle": (245, 172, 147),
    "left seminal vesicle": (245, 172, 147),
    "right deferent duct": (241, 172, 151),
    "left deferent duct": (241, 172, 151),
    "skin of abdominal wall": (177, 124, 92),
    "muscles of abdominal wall": (171, 85, 68),
    "skeleton of abdomen": (217, 198, 131),
    "lumbar vertebral column": (212, 188, 102),
    "female external genitalia": (185, 135, 134),
    "male external genitalia": (185, 135, 134),
    "skeleton of upper limb": (198, 175, 125),
    "muscles of upper limb": (194, 98, 79),
    "right upper limb": (177, 122, 101),
    "left upper limb": (177, 122, 101),
    "right shoulder": (177, 122, 101),
    "left shoulder": (177, 122, 101),
    "right arm": (177, 122, 101),
    "left arm": (177, 122, 101),
    "right elbow": (177, 122, 101),
    "left elbow": (177, 122, 101),
    "right forearm": (177, 122, 101),
    "left forearm": (177, 122, 101),
    "right wrist": (177, 122, 101),
    "left wrist": (177, 122, 101),
    "right hand": (177, 122, 101),
    "left hand": (177, 122, 101),
    "skeleton of lower limb": (255, 238, 170),
    "muscles of lower limb": (206, 111, 93),
    "right lower limb": (177, 122, 101),
    "left lower limb": (177, 122, 101),
    "right hip": (177, 122, 101),
    "left hip": (177, 122, 101),
    "right thigh": (177, 122, 101),
    "left thigh": (177, 122, 101),
    "right knee": (177, 122, 101),
    "left knee": (177, 122, 101),
    "right leg": (177, 122, 101),
    "left leg": (177, 122, 101),
    "right foot": (177, 122, 101),
    "left foot": (177, 122, 101),
    "peripheral nervous system": (216, 186, 0),
    "autonomic nerve": (255, 226, 77),
    "sympathetic trunk": (255, 243, 106),
    "cranial nerves": (255, 234, 92),
    "vagus nerve": (240, 210, 35),
    "peripheral nerve": (224, 194, 0),
    "circulatory system": (213, 99, 79),
    "systemic arterial system": (217, 102, 81),
    "systemic venous system": (0, 147, 202),
    "pulmonary arterial system": (0, 122, 171),
    "pulmonary venous system": (186, 77, 64),
    "lymphatic system": (111, 197, 131),
    "needle": (240, 255, 30),
    "region 0": (185, 232, 61),
    "region 1": (0, 226, 255),
    "region 2": (251, 159, 255),
    "region 3": (230, 169, 29),
    "region 4": (0, 194, 113),
    "region 5": (104, 160, 249),
    "region 6": (221, 108, 158),
    "region 7": (137, 142, 0),
    "region 8": (230, 70, 0),
    "region 9": (0, 147, 0),
    "region 10": (0, 147, 248),
    "region 11": (231, 0, 206),
    "region 12": (129, 78, 0),
    "region 13": (0, 116, 0),
    "region 14": (0, 0, 255),
    "region 15": (157, 0, 0),
    "unknown": (100, 100, 130),
    "cyst": (205, 205, 100),
}
