import cgi
import copy
import http.client as httplib
import json
import logging
import mimetypes
import os
import shutil
import tempfile
import time
import traceback
from collections import OrderedDict
from urllib.parse import quote_plus
from urllib.parse import urlparse

import SampleData
import ctk
import qt
import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


class MONAILabel(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MONAI Label"
        self.parent.categories = ["Active Learning"]
        self.parent.dependencies = []
        self.parent.contributors = ["NVIDIA, KCL"]
        self.parent.helpText = """
Active Learning solution.
See more information in <a href="https://github.com/MONAI/MONAI-Label">module documentation</a>.
"""
        self.parent.acknowledgementText = """
Developed by NVIDIA, KCL
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", self.initializeAfterStartup)

    def initializeAfterStartup(self):
        if not slicer.app.commandOptions().noMainWindow:
            self.settingsPanel = MONAILabelSettingsPanel()
            slicer.app.settingsDialog().addPanel("MONAI-Label", self.settingsPanel)


class _ui_MONAILabelSettingsPanel(object):
    def __init__(self, parent):
        vBoxLayout = qt.QVBoxLayout(parent)

        # settings
        groupBox = ctk.ctkCollapsibleGroupBox()
        groupBox.title = "MONAI Label Server"
        groupLayout = qt.QFormLayout(groupBox)

        serverUrl = qt.QLineEdit()
        groupLayout.addRow("Server address:", serverUrl)
        parent.registerProperty(
            "MONAI-Label/serverUrl", serverUrl,
            "text", str(qt.SIGNAL("textChanged(QString)")))

        serverUrlHistory = qt.QLineEdit()
        groupLayout.addRow("Server address history:", serverUrlHistory)
        parent.registerProperty(
            "MONAI-Label/serverUrlHistory", serverUrlHistory,
            "text", str(qt.SIGNAL("textChanged(QString)")))

        useSessionCheckBox = qt.QCheckBox()
        useSessionCheckBox.checked = False
        useSessionCheckBox.toolTip = (
            "Enable this option to make use of sessions while bringing any external image."
            " Volume is uploaded to MONAI Label as part of session once and it makes inference operations faster.")
        groupLayout.addRow("Session:", useSessionCheckBox)
        useSessionMapper = ctk.ctkBooleanMapper(useSessionCheckBox, "checked", str(qt.SIGNAL("toggled(bool)")))
        parent.registerProperty(
            "MONAI-Label/session", useSessionMapper,
            "valueAsInt", str(qt.SIGNAL("valueAsIntChanged(int)")))

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
        self._updatingGUIFromParameterNode = False

        self.models = OrderedDict()
        self.config = OrderedDict()

        self.progressBar = None

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/MONAILabel.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MONAILabelLogic()

        # Set icons and tune widget properties
        self.ui.serverComboBox.lineEdit().setPlaceholderText("enter server address or leave empty to use default")
        self.ui.fetchModelsButton.setIcon(self.icon('refresh-icon.png'))
        self.ui.segmentationButton.setIcon(self.icon('segment.png'))
        self.ui.nextSampleButton.setIcon(self.icon('segment.png'))
        self.ui.saveLabelButton.setIcon(self.icon('save.png'))

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
        self.ui.fetchModelsButton.connect('clicked(bool)', self.onClickFetchModels)
        self.ui.serverComboBox.connect('currentIndexChanged(int)', self.onClickFetchModels)
        self.ui.segmentationModelSelector.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.segmentationButton.connect('clicked(bool)', self.onClickSegmentation)
        self.ui.deepgrowModelSelector.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.nextSampleButton.connect('clicked(bool)', self.onNextSampleButton)

        self.initializeParameterNode()

    def cleanup(self):
        self.removeObservers()

    def enter(self):
        self.initializeParameterNode()

    def exit(self):
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        self.setParameterNode(None)

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

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.updateSelector(self.ui.segmentationModelSelector, ['segmentation'], 'SegmentationModel', 0)
        self.updateSelector(self.ui.deepgrowModelSelector, ['deepgrow'], 'DeepgrowModel', 0)
        self.updateConfigTable(self.ui.configTable)

        # Enable/Disable
        self.ui.segmentationButton.setEnabled(self.ui.segmentationModelSelector.currentText)

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

        self._parameterNode.EndModify(wasModified)

    def updateSelector(self, selector, model_types, param, defaultIndex=0):
        wasSelectorBlocked = selector.blockSignals(True)
        selector.clear()

        for model_name, model in self.models.items():
            if model['type'] in model_types:
                selector.addItem(model_name)
                selector.setItemData(selector.count - 1, model['description'], qt.Qt.ToolTipRole)

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

    def updateConfigTable(self, table):
        table.clear()
        table.setHorizontalHeaderLabels(['section', 'name', 'value'])

        config = copy.deepcopy(self.config)
        train = config.get('train', {})
        activelearning = config.get('activelearning', {})
        if train:
            config.pop('train')
        if activelearning:
            config.pop('activelearning')

        table.setRowCount(len(config) + len(activelearning) + len(train))
        config = {"all": config, "activelearning": activelearning, "train": train}
        colors = {"all": qt.QColor(255, 255, 255), "activelearning": qt.QColor(220, 220, 220),
                  "train": qt.QColor(255, 255, 255)}

        n = 0
        for section in config:
            for key in config[section]:
                table.setSpan(n, 0, n + len(config[section]) - 1, 1)
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
                    combo = qt.QComboBox()
                    combo.addItem('true')
                    combo.addItem('false')
                    combo.setCurrentIndex(0 if val else 1)
                    table.setCellWidget(n, 2, combo)
                else:
                    table.setItem(n, 2, qt.QTableWidgetItem(str(val)))

                table.item(n, 0).setBackground(colors[section])
                n = n + 1

    def icon(self, name='MONAILabel.png'):
        # It should not be necessary to modify this method
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", name)
        if os.path.exists(iconPath):
            return qt.QIcon(iconPath)
        return qt.QIcon()

    def updateServerSettings(self):
        self.logic.setServer(self.serverUrl())
        self.logic.setUseSession(slicer.util.settingsValue(
            "MONAI-Label/session",
            False, converter=slicer.util.toBool))

        self.saveServerUrl()

    def serverUrl(self):
        serverUrl = self.ui.serverComboBox.currentText
        if not serverUrl:
            serverUrl = "http://127.0.0.1:8000"
        return serverUrl

    def saveServerUrl(self):
        self.updateParameterNodeFromGUI()

        # Save selected server URL
        settings = qt.QSettings()
        serverUrl = self.ui.serverComboBox.currentText
        settings.setValue("MONAI-Label/serverUrl", serverUrl)

        # Save current server URL to the top of history
        serverUrlHistory = settings.value("MONAI-Label/serverUrlHistory")
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
        settings.setValue("MONAI-Label/serverUrlHistory", ";".join(serverUrlHistory))

        self.updateServerUrlGUIFromSettings()

    def onClickFetchModels(self):
        self.fetchModels(showInfo=False)

    def fetchModels(self, showInfo=False):
        if not self.logic:
            return

        start = time.time()
        try:
            self.updateServerSettings()
            info = self.logic.info()
            models = info["models"]
        except:
            slicer.util.errorDisplay("Failed to fetch models from remote server. "
                                     "Make sure server address is correct and <server_uri>/info/ "
                                     "is accessible in browser",
                                     detailedText=traceback.format_exc())
            return

        self.models.clear()
        self.config = info["config"]
        model_count = {}
        for k, v in models.items():
            model_type = v.get('type', 'segmentation')
            model_count[model_type] = model_count.get(model_type, 0) + 1

            logging.debug('{} = {}'.format(k, model_type))
            self.models[k] = v

        self.updateGUIFromParameterNode()

        msg = ''
        msg += '-----------------------------------------------------\t\n'
        msg += 'Total Models Available: \t' + str(len(models)) + '\t\n'
        msg += '-----------------------------------------------------\t\n'
        for model_type in model_count.keys():
            msg += model_type.capitalize() + ' Models: \t' + str(model_count[model_type]) + '\t\n'
        msg += '-----------------------------------------------------\t\n'

        if showInfo:
            qt.QMessageBox.information(slicer.util.mainWindow(), 'MONAI Label', msg)
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

    def onNextSampleButton(self):
        if not self.logic:
            return

        start = time.time()
        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

            self.updateServerSettings()
            sample = self.logic.next_sample()
            logging.debug(sample)

            image_file = sample["image"].replace("/workspace", "/raid/sachi")
            print(image_file)
            if os.path.exists(image_file):
                slicer.util.loadVolume(image_file)
            else:
                download_uri = f"{self.serverUrl()}{sample['url']}"
                logging.info(download_uri)

                sampleDataLogic = SampleData.SampleDataLogic()
                sampleDataLogic.downloadFromURL(
                    nodeNames=sample["name"],
                    fileNames=sample["name"],
                    uris=download_uri,
                    checksums=sample["checksum"])
        except:
            slicer.util.errorDisplay("Failed to fetch Sample from MONAI Label Server",
                                     detailedText=traceback.format_exc())
        finally:
            qt.QApplication.restoreOverrideCursor()

        logging.info("Time consumed by next_sample: {0:3.1f}".format(time.time() - start))

    def onClickSegmentation(self):
        pass

    def onClickDeepgrow(self, current_point):
        pass

    def createCursor(self, widget):
        return slicer.util.mainWindow().cursor

    def updateServerUrlGUIFromSettings(self):
        # Save current server URL to the top of history
        settings = qt.QSettings()
        serverUrlHistory = settings.value("MONAI-Label/serverUrlHistory")
        wasBlocked = self.ui.serverComboBox.blockSignals(True)
        self.ui.serverComboBox.clear()
        if serverUrlHistory:
            self.ui.serverComboBox.addItems(serverUrlHistory.split(";"))
        self.ui.serverComboBox.setCurrentText(settings.value("MONAI-Label/serverUrl"))
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
    def __init__(self, server_url=None, progress_callback=None):
        ScriptedLoadableModuleLogic.__init__(self)

        self.tmpdir = slicer.util.tempDirectory('slicer-monai-label')
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

    def __del__(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def inputFileExtension(self):
        return ".nii.gz" if self.useCompression else ".nii"

    def outputFileExtension(self):
        return ".nii.gz"

    def setServer(self, server_url=None):
        if not server_url:
            server_url = "http://127.0.0.1:8000"
        self.server_url = server_url

    def setUseSession(self, useSession):
        self.useSession = useSession

    def setProgressCallback(self, progress_callback=None):
        self.progress_callback = progress_callback

    def reportProgress(self, progress):
        if self.progress_callback:
            self.progress_callback(progress)

    def info(self):
        return MONAILabelClient(self.server_url).info()

    def next_sample(self):
        return MONAILabelClient(self.server_url).next_sample()

    def inference(self, image_in, model, params=None):
        logging.debug('Preparing input data for segmentation')
        self.reportProgress(0)

        result_file = tempfile.NamedTemporaryFile(suffix=self.outputFileExtension(), dir=self.tmpdir).name
        client = MONAILabelClient(self.server_url)
        params = client.inference(
            model=model,
            params=params,
            image_in=image_in,
            image_out=result_file
        )

        logging.debug('Extreme Points: {}'.format(params))

        self.reportProgress(100)
        return result_file, params


class MONAILabelClient:
    def __init__(self, server_url='http://127.0.0.1:8000'):
        self._server_url = server_url

    def get_server_url(self):
        return self._server_url

    def set_server_url(self, server_url):
        self._server_url = server_url

    def info(self):
        selector = '/info/'
        status, response = MONAILabelUtils.http_method('GET', self._server_url, selector)
        if status != 200:
            raise MONAILabelException(MONAILabelError.SERVER_ERROR, 'Status: {}; Response: {}'.format(status, response))

        response = response.decode('utf-8') if isinstance(response, bytes) else response
        logging.debug('Response: {}'.format(response))
        return json.loads(response)

    def next_sample(self, strategy="random"):
        selector = '/activelearning/next_sample'
        body = {'strategy': strategy}
        status, response = MONAILabelUtils.http_method('POST', self._server_url, selector, body)
        if status != 200:
            raise MONAILabelException(MONAILabelError.SERVER_ERROR, 'Status: {}; Response: {}'.format(status, response))

        response = response.decode('utf-8') if isinstance(response, bytes) else response
        logging.debug('Response: {}'.format(response))
        return json.loads(response)

    def inference(self, model, params, image_in, image_out):
        selector = '/infer/{}?image={}'.format(
            MONAILabelUtils.urllib_quote_plus(model),
            MONAILabelUtils.urllib_quote_plus(image_in))
        in_fields = {'params': params}

        status, form, files = MONAILabelUtils.http_multipart('POST', self._server_url, selector, in_fields, {})
        if status != 200:
            raise MONAILabelException(MONAILabelError.SERVER_ERROR, 'Status: {}; Response: {}'.format(status, form))

        form = json.loads(form) if isinstance(form, str) else form
        params = form.get('params') if files else form
        params = json.loads(params) if isinstance(params, str) else params

        MONAILabelUtils.save_result(files, image_out)
        return image_out, params


class MONAILabelError:
    RESULT_NOT_FOUND = 1
    SERVER_ERROR = 2
    UNKNOWN = 3


class MONAILabelException(Exception):
    def __init__(self, error, msg):
        self.error = error
        self.msg = msg


class MONAILabelUtils:
    @staticmethod
    def http_method(method, server_url, selector, body=None):
        logging.debug('{} {}{}'.format(method, server_url, selector))

        parsed = urlparse(server_url)
        conn = httplib.HTTPConnection(parsed.hostname, parsed.port)

        path = parsed.path.rstrip('/')
        selector = path + '/' + selector.lstrip('/')
        logging.debug('URI Path: {}'.format(selector))

        conn.request(method, selector, body=json.dumps(body) if body else None)
        response = conn.getresponse()

        logging.debug('HTTP Response Code: {}'.format(response.status))
        logging.debug('HTTP Response Message: {}'.format(response.reason))
        logging.debug('HTTP Response Headers: {}'.format(response.getheaders()))
        return response.status, response.read()

    @staticmethod
    def http_multipart(method, server_url, selector, fields, files):
        logging.debug('{} {}{}'.format(method, server_url, selector))

        parsed = urlparse(server_url)
        conn = httplib.HTTPConnection(parsed.hostname, parsed.port)

        content_type, body = MONAILabelUtils.encode_multipart_formdata(fields, files)
        headers = {'content-type': content_type, 'content-length': str(len(body))}

        path = parsed.path.rstrip('/')
        selector = path + '/' + selector.lstrip('/')
        logging.debug('URI Path: {}'.format(selector))

        conn.request(method, selector, body, headers)

        response = conn.getresponse()
        logging.debug('HTTP Response Code: {}'.format(response.status))
        logging.debug('HTTP Response Message: {}'.format(response.reason))
        logging.debug('HTTP Response Headers: {}'.format(response.getheaders()))

        response_content_type = response.getheader('content-type', content_type)
        logging.debug('HTTP Response Content-Type: {}'.format(response_content_type))

        if 'multipart' in response_content_type:
            if response.status == 200:
                form, files = MONAILabelUtils.parse_multipart(response.fp if response.fp else response, response.msg)
                logging.debug('Response FORM: {}'.format(form))
                logging.debug('Response FILES: {}'.format(files.keys()))
                return response.status, form, files
            else:
                return response.status, response.read(), None

        logging.debug('Reading status/content from simple response!')
        return response.status, response.read(), None

    @staticmethod
    def save_result(files, result_file):
        if result_file is None:
            return

        if len(files) == 0:
            raise MONAILabelException(MONAILabelError.RESULT_NOT_FOUND, "No result files found in server response!")

        for name in files:
            data = files[name]
            logging.debug('Saving {} to {}; Size: {}'.format(name, result_file, len(data)))

            dir_path = os.path.dirname(os.path.realpath(result_file))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            with open(result_file, "wb") as f:
                if isinstance(data, bytes):
                    f.write(data)
                else:
                    f.write(data.encode('utf-8'))
            break

    @staticmethod
    def encode_multipart_formdata(fields, files):
        limit = '----------lImIt_of_THE_fIle_eW_$'
        lines = []
        for (key, value) in fields.items():
            lines.append('--' + limit)
            lines.append('Content-Disposition: form-data; name="%s"' % key)
            lines.append('')
            lines.append(value)
        for (key, filename) in files.items():
            lines.append('--' + limit)
            lines.append('Content-Disposition: form-data; name="%s"; filename="%s"' % (key, filename))
            lines.append('Content-Type: %s' % MONAILabelUtils.get_content_type(filename))
            lines.append('')
            with open(filename, mode='rb') as f:
                data = f.read()
                lines.append(data)
        lines.append('--' + limit + '--')
        lines.append('')

        body = bytearray()
        for line in lines:
            body.extend(line if isinstance(line, bytes) else line.encode('utf-8'))
            body.extend(b'\r\n')

        content_type = 'multipart/form-data; boundary=%s' % limit
        return content_type, body

    @staticmethod
    def get_content_type(filename):
        return mimetypes.guess_type(filename)[0] or 'application/octet-stream'

    @staticmethod
    def parse_multipart(fp, headers):
        logger = logging.getLogger(__name__)
        fs = cgi.FieldStorage(
            fp=fp,
            environ={'REQUEST_METHOD': 'POST'},
            headers=headers,
            keep_blank_values=True
        )
        form = {}
        files = {}
        if hasattr(fs, 'list') and isinstance(fs.list, list):
            for f in fs.list:
                logger.debug('FILE-NAME: {}; NAME: {}; SIZE: {}'.format(f.filename, f.name, len(f.value)))
                if f.filename:
                    files[f.filename] = f.value
                else:
                    form[f.name] = f.value
        return form, files

    # noinspection PyUnresolvedReferences
    @staticmethod
    def urllib_quote_plus(s):
        return quote_plus(s)


class MONAILabelTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_MONAILabel1()

    def test_MONAILabel1(self):
        self.delayDisplay('Test passed')
