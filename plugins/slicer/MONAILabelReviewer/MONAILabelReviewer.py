from code import interact
from typing import Dict, List
import datetime
import logging

import qt, slicer
import qSlicerSegmentationsModuleWidgetsPythonQt
from slicer.ScriptedLoadableModule import *

from slicer.util import VTKObservationMixin
import re

from ReviewerLibs.MONAILabelReviewerLogic import MONAILabelReviewerLogic
from ReviewerLibs.ImageData import ImageData
from ReviewerLibs.MONAILabelReviewerEnum import SegStatus
from ReviewerLibs.MONAILabelReviewerEnum import Level


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
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False


    self.STATUS = SegStatus()
    self.LEVEL = Level()

    self.selectedReviewer : str = ""
    self.selectedClientId : str = ""
    self.listImageData: List[ImageData] = None
    self.imageCounter : int = 0
    self.currentImageData : ImageData = None
    self.idToimageData : Dict[str, ImageData]= None

    #Meta Information
    self.finalStatus : str=""
    self.finalLevel : str=""
    self.finalComment : str=""

    self.reviewersModeIsActive = False

    self.mapFiltersToBool : Dict[str, bool] = {"segmented" : False,
                                              "notSegemented" : False,
                                              "approved" : False,
                                              "flagged" : False}

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/MONAILabelReviewer.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    self.dialogWidget = slicer.util.loadUI(self.resourcePath('UI/DialogInfo.ui'))
    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = MONAILabelReviewerLogic()

    self.setLightVersion()
    self.segmentEditorWidget = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
    self.addSegmentator()
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
    self.dialogWidget.pushButton.clicked.connect(self.closeDialog)
    self.ui.comboBox_clients.currentIndexChanged.connect(self.index_changed)
    self.ui.comboBox_reviewers.currentIndexChanged.connect(self.indexReviewerchanged)

  def getCurrentTime(self):
    return datetime.datetime.now()

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def indexReviewerchanged(self, index):
    logging.info("{}: Selected reviewer: '{}'".format(self.getCurrentTime(), self.ui.comboBox_reviewers.currentText))
    self.selectedReviewer = self.ui.comboBox_reviewers.currentText

  def index_changed(self, index):
    self.loadImageData()

#Section: Dialog window
  def displayDialogInfo(self, title : str, message : str):
    self.dialogWidget.label_title.setText(title)
    self.dialogWidget.label_message.setText(message)
    self.dialogWidget.exec()

  def closeDialog(self):
    self.dialogWidget.close()

  def setReviewerVersion(self):
    self.reviewersModeIsActive = True
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

    #imag information
    self.ui.label_14.show()
    self.ui.lineEdit_status.show()
    self.ui.label_16.show()
    self.ui.lineEdit_level.show()
    self.ui.plainText_comment.show()
    if(self.ui.btn_basic_mode.isChecked()):
      self.ui.btn_basic_mode.setChecked(False)

    self.collapseAllSecions()

#Section:  Light version Option

  def setLightVersion(self):
    self.reviewersModeIsActive = False
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

    #imag information
    self.ui.label_14.hide()
    self.ui.lineEdit_status.hide()
    self.ui.label_16.hide()
    self.ui.lineEdit_level.hide()
    self.ui.plainText_comment.hide()
    if(self.ui.btn_reviewers_mode.isChecked()):
      self.ui.btn_reviewers_mode.setChecked(False)

    self.collapseAllSecions()

  def cleanCache(self):
    self.logic = MONAILabelReviewerLogic()
    self.selectedReviewer = ""
    self.selectedClientId = ""
    self.listImageData = None
    self.imageCounter = 0
    self.currentImageData = None
    self.idToimageData = None

    #Meta Information
    self.finalStatus = ""
    self.finalLevel = ""
    self.finalComment = ""
    logging.info("{}: Cache is cleaned".format(self.getCurrentTime()))
  

#Section: Server
  def loadServerSelection(self):
    settings = qt.QSettings()
    serverUrlHistory = settings.value("MONAILabel/serverUrlHistory")
    
    self.ui.comboBox_server_url.clear()
    self.ui.comboBox_server_url.addItems(serverUrlHistory.split(";"))

  def init_dicom_stream(self):
    '''
    initiates connection to monai server
    Default: client listens on "http://127.0.0.1:8000"
    '''
    #Check Connection
    self.cleanCache()
    serverUrl : str = self.ui.comboBox_server_url.currentText
    isConnected : bool = self.logic.connectToMonaiServer(serverUrl)
    if(not isConnected):
      warningMessage = "Connection to server failed \ndue to invalid ip '{}'".format(serverUrl)
      self.displayDialogInfo(title="Warning", message=warningMessage)
      return
    result = self.logic.initMetaDataProcessing()
    if(result==False):
      warningMessage = "Request for datastore-info failed.\nPlease check if server address is correct \n('{}')!".format(serverUrl)
      self.displayDialogInfo(title="Warning", message=warningMessage)
      return
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
    if(self.reviewersModeIsActive and self.selectedReviewer == ""):
      warningMessage = "Missing reviewer's name.\nPlease enter your id or name in the reviewer's field!"
      logging.warning(warningMessage)
      self.displayDialogInfo(title="Warning", message=warningMessage)
      return
    self.ui.collapsibleButton_search_image.enabled = True
    self.ui.collapsibleButton_dicom_stream.enabled = True
    
    #set Segmentation progress bar 
    self.setProgessBar()

    #fill combobox
    self.fillComboBoxes()

    #set up buttons
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
    self.ui.btn_show_image.enabled = False

  def setProgessBar(self):
    progessBarData = self.logic.getStatistics()
    self.ui.progressBar_segmentation.setProperty("value", progessBarData['segmentationProgress'])
    self.ui.label_idx_seg_image.setText(progessBarData['idxTotalSegmented'])
    self.ui.label_idx_appr_image.setText(progessBarData['idxTotalApproved'])
    self.ui.progressBar_approved_total.setProperty("value", progessBarData['progressPercentage'])

  def fillComboBoxes(self):
    #clients
    clientIds = self.logic.getClientIds()
    
    self.ui.comboBox_clients.clear()
    self.ui.comboBox_clients.addItem("All")
    for clientId in clientIds:
      self.ui.comboBox_clients.addItem(str(clientId))

    #combobox in search section
    self.ui.comboBox_search_annotator.clear()
    self.ui.comboBox_search_annotator.addItem("All")
    for clientId in clientIds:
      self.ui.comboBox_search_annotator.addItem(str(clientId))

    #reviewers
    reviewers = self.logic.getReviewers()
    self.ui.comboBox_reviewers.addItem(self.selectedReviewer)
    for reviewer in reviewers:
      if(reviewer == self.selectedReviewer):
        continue
      self.ui.comboBox_reviewers.addItem(str(reviewer))

    #combobox in search section
    self.ui.comboBox_search_reviewer.addItem("All")
    for reviewer in reviewers:
      self.ui.comboBox_search_reviewer.addItem(str(reviewer))

  def cleanDicomStreamSection(self):
    self.finalStatus = ""
    self.finalLevel = ""
    self.finalComment = ""

    self.selectedClientId = None
    self.imageCounter = 0
    self.currentImageData = None
    self.idToimageData = None
    self.listImageData = None

    self.cleanProgressBarDicomStreamSection()
    self.cleanCheckBoxes()
    self.resetHorizontalSlider()


#Section: Loading images
  def loadImageData(self):
    if((self.selectedClientId == self.ui.comboBox_clients.currentText) 
        and (self.isDifferentFilter() == False)):
      return
    self.imageCounter = 0

    self.cleanSearchSection()
    #select segmentator: ALL
    self.selectedClientId = self.ui.comboBox_clients.currentText
    if(self.selectedClientId == "All"):
      self.listImageData = self.loadImageDataWithFilter(selectedClientId = "")
      self.ui.checkBox_segmented.setEnabled(True)
      self.ui.checkBox_not_segmented.setEnabled(True)
      self.setProgressBarOfAll()
    
    #select segmentator: client was selected
    if(self.selectedClientId != "All"):
      self.listImageData = self.loadImageDataWithFilter(selectedClientId = self.selectedClientId)
      self.setCheckBoxesClient()
      self.setProgressBarOfClient(self.selectedClientId)
    
    logging.info("{}: Successfully loaded Image data [total = {}, category = '{}']".format(self.getCurrentTime(), len(self.listImageData), self.selectedClientId))
    
    if(len(self.listImageData)>0):
      self.currentImageData = self.listImageData[self.imageCounter]
      self.loadNextImage(self.currentImageData)
    
    self.ui.collapsibleButton_dicom_evaluation.enabled = True
    self.ui.collapsibleButton_dicom_evaluation.collapsed = False
    self.setHorizontalSlider(len(self.listImageData))
    self.collectFilters()
    self.setLoadButtonColor(reload = False)

  def loadImageDataWithFilter(self, selectedClientId : str) -> list:
    isApproved = bool(self.ui.checkBox_approved.isChecked())
    isFlagged =  bool(self.ui.checkBox_flagged.isChecked())
    isNotSegmented = bool(self.ui.checkBox_not_segmented.isChecked())
    segmented = bool(self.ui.checkBox_segmented.isChecked())
    logging.info("{}: Selected filters: segmented= {} | isNotSegmented= {} | isApproved= {} | isFlagged= {}".format(self.getCurrentTime(), segmented, isNotSegmented, isApproved, isFlagged))
    if(selectedClientId==""):
      return self.logic.getAllImageData(segmented, isNotSegmented, isApproved, isFlagged)
    return self.logic.getImageDataByClientId(selectedClientId, isApproved, isFlagged)


  def setProgressBarOfAll(self):
    progessBarData = self.logic.getStatistics()
    # Progress bar: Segmented/TotalImage
    self.ui.progressBar_segmented_client.setProperty("value",  progessBarData["segmentationProgressAllPercentage"])
    self.ui.label_idx_seg_image_client.setText(progessBarData["idxTotalSegmented"])
    # Progress bar: approvalCount/TotalImage
    self.ui.progressBar_approved_client.setProperty("value", progessBarData["approvalProgressPercentage"])
    self.ui.label_idx_appr_image_client.setText(progessBarData["idxTotalApproved"])

  def cleanProgressBarDicomStreamSection(self):
    self.ui.progressBar_segmented_client.setProperty("value", 0)
    self.ui.progressBar_approved_client.setProperty("value", 0)
    self.ui.label_idx_seg_image_client.setText("x/y")
    self.ui.label_idx_appr_image_client.setText("x/y")

  def setLoadButtonColor(self, reload : bool):
    if(reload): #reload required
       self.ui.btn_load.setStyleSheet("background-color : rgb(169, 169, 169)")
       return
    self.ui.btn_load.setStyleSheet("background-color : rgb(0, 144, 81)")

  def setProgressBarOfClient(self, selectedClientId : str):
    percentageApprovedOfClient, idxApprovedOfClient = self.logic.getPercentageApproved(selectedClientId)  
    self.ui.progressBar_approved_client.setProperty("value", percentageApprovedOfClient)
    self.ui.label_idx_appr_image_client.setText(idxApprovedOfClient)

    percentageSemgmentedByClient,  idxSegmentedByClient= self.logic.getPercentageSemgmentedByClient(selectedClientId)
    self.ui.progressBar_segmented_client.setProperty("value", percentageSemgmentedByClient)
    self.ui.label_idx_seg_image_client.setText(idxSegmentedByClient)

  def setHorizontalSlider(self, loadesImageCount : int):
    self.ui.horizontalSlider_image_idx.setMinimum(0)
    self.ui.horizontalSlider_image_idx.setMaximum(loadesImageCount-1)
    idxImage = "Image: {}/{}".format(self.imageCounter + 1, len(self.listImageData))
    self.ui.label_idx_image.setText(idxImage)

  def updateHorizontalSlider(self):
    self.ui.horizontalSlider_image_idx.setValue(self.imageCounter)
    idxImage = "Image: {}/{}".format(self.imageCounter+ 1, len(self.listImageData))
    self.ui.label_idx_image.setText(idxImage)

  def resetHorizontalSlider(self):
    self.ui.horizontalSlider_image_idx.setValue(1)
    self.ui.label_idx_image.setText("Image:")


#Section: Filter   
  def collectFilters(self):
    self.mapFiltersToBool["segmented"] = self.ui.checkBox_segmented.isChecked()
    self.mapFiltersToBool["notSegemented"] = self.ui.checkBox_not_segmented.isChecked()
    self.mapFiltersToBool["approved"] = self.ui.checkBox_approved.isChecked()
    self.mapFiltersToBool["flagged"] = self.ui.checkBox_flagged.isChecked()

  def isDifferentFilter(self) -> bool:
    if(self.mapFiltersToBool["segmented"] != self.ui.checkBox_segmented.isChecked()):
      return True
    if(self.mapFiltersToBool["notSegemented"] != self.ui.checkBox_not_segmented.isChecked()):
      return True
    if(self.mapFiltersToBool["approved"] != self.ui.checkBox_approved.isChecked()):
      return True
    if(self.mapFiltersToBool["flagged"] != self.ui.checkBox_flagged.isChecked()):
      return True
    return False

  #CheckBox: clean
  def cleanCheckBoxes(self):
      self.ui.checkBox_segmented.setChecked(False)
      self.ui.checkBox_not_segmented.setChecked(False)
      self.ui.checkBox_flagged.setChecked(False)
      self.ui.checkBox_approved.setChecked(False)

  #CheckBox: flagged
  def setCheckBoxesClient(self):
      self.setLoadButtonColor(reload = True)
      self.ui.checkBox_not_segmented.setEnabled(False)
      self.ui.checkBox_segmented.setChecked(True)
      self.ui.checkBox_segmented.setEnabled(False)

  #CheckBox: flagged
  def checkedFlagged(self):
    self.setLoadButtonColor(reload = True)
    self.ui.checkBox_segmented.setChecked(True)
    if(self.ui.checkBox_approved.isChecked()):
      self.ui.checkBox_approved.setChecked(False)
    if(self.ui.checkBox_not_segmented.isChecked()):
      self.ui.checkBox_not_segmented.setChecked(False)
          
  #CheckBox: approved
  def checkApproved(self):
    self.setLoadButtonColor(reload = True)
    self.ui.checkBox_segmented.setChecked(True)
    if(self.ui.checkBox_flagged.isChecked()):
      self.ui.checkBox_flagged.setChecked(False)
    if(self.ui.checkBox_not_segmented.isChecked()):
      self.ui.checkBox_not_segmented.setChecked(False)

  #CheckBox: NOT segmented
  def checkNotSegmented(self):
    self.setLoadButtonColor(reload = True)
    if(self.ui.checkBox_approved.isChecked()):
      self.ui.checkBox_approved.setChecked(False)
    if(self.ui.checkBox_flagged.isChecked()):
      self.ui.checkBox_flagged.setChecked(False)
    if(self.ui.checkBox_segmented.isChecked()):
      self.ui.checkBox_segmented.setChecked(False)

  #CheckBox: segmented
  def checkSegmented(self):
    self.setLoadButtonColor(reload = True)
    if(self.ui.checkBox_segmented.isChecked() == False):
      self.ui.checkBox_approved.setChecked(False)
      self.ui.checkBox_flagged.setChecked(False)
      return

    if(self.ui.checkBox_not_segmented.isChecked()):
      self.ui.checkBox_not_segmented.setChecked(False)

  
#Section: Search Image
  def cleanSearchSection(self):
    self.ui.tableWidge_imageMeta.setRowCount(0)
    self.ui.tableWidge_imageMeta.clearContents()
    self.ui.textEdit_search.clear()

  def search(self):
    '''
    After triggering search button, load images and segmentation by input ids
    '''
    self.cleanDicomStreamSection()

    if(self.ui.textEdit_search.toPlainText() == ""):
      logging.info("{}: Search input field is empty".format(self.getCurrentTime()))
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
    if(len(foundIdList)>0):
      self.setSearchResultMessage(numOfFound = len(foundIdList))
      self.loadFirstImage()
    else:
      self.setSearchResultMessage(numOfFound = 0)

  def searchByAnnotatorReviewer(self):
    selectedAnnotator : str = self.ui.comboBox_search_annotator.currentText
    selectedReviewer : str = self.ui.comboBox_search_reviewer.currentText
    isApproved : bool = bool(self.ui.checkBox_search_approved.isChecked())
    isFlagged : bool =  bool(self.ui.checkBox_search_flagged.isChecked())
    
    self.idToimageData = self.logic.searchByAnnotatorReviewer(selectedAnnotator, selectedReviewer, isApproved, isFlagged)
    self.listImageData = [*self.idToimageData.values()]
    
    self.loadSearchImageMetaInTable(self.listImageData, [])
    if(len(self.listImageData)>0):
      self.ui.collapsibleButton_dicom_evaluation.enabled = True
      self.setSearchResultMessage(numOfFound = len(self.idToimageData))
      self.setHorizontalSlider(len(self.idToimageData))
      self.loadFirstImage()
    else:
      self.setSearchResultMessage(numOfFound = 0)

  def searchByLevel(self):
      isEasy : bool = bool(self.ui.checkBox_search_easy.isChecked())
      isMedium : bool =  bool(self.ui.checkBox_search_medium.isChecked())
      isHard : bool =  bool(self.ui.checkBox_search_hard.isChecked())

      self.idToimageData = self.logic.searchByLevel(isEasy, isMedium, isHard)
      self.listImageData = [*self.idToimageData.values()]

      self.loadSearchImageMetaInTable(self.listImageData, [])
      if(len(self.listImageData)>0):
        self.ui.collapsibleButton_dicom_evaluation.enabled = True
        self.setSearchResultMessage(numOfFound = len(self.idToimageData))
        self.setHorizontalSlider(len(self.idToimageData))
        self.loadFirstImage()
      else:
        self.setSearchResultMessage(numOfFound = 0)

      

  def setSearchResultMessage(self, numOfFound : int):
    if(numOfFound==0):
      self.ui.label_search_result.setText("Result: No images found.")
      self.ui.label_search_result.setStyleSheet("color: red")
    else:
      resultMessage = "Result: {} images found.".format(numOfFound)
      self.ui.label_search_result.setText(resultMessage)
      self.ui.label_search_result.setStyleSheet("color: green")


  def checkedAppprovedSearch(self):
    isFlagged : bool = bool(self.ui.checkBox_search_flagged.isChecked())
    if(isFlagged):
      self.ui.checkBox_search_flagged.setChecked(False)

  def checkedFlaggedSearch(self):
    isApproved : bool = bool(self.ui.checkBox_search_approved.isChecked())
    if(isApproved):
      self.ui.checkBox_search_approved.setChecked(False)
  

  def loadSearchImageMetaInTable(self, foundlist : List[ImageData], notFoundIdList : List[str]):
    '''
    Set table content after triggering button "show"
    Parameters:
        foundlist (list): list contains found ids
        notFoundIdList (list): list contains not found ids
    '''
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
    #self.imageCounter += 1
    self.updateHorizontalSlider()

  def showSearchedImage(self):
    '''
    displays dicom & segmentation to corresponding selected row in listed ids
    '''
    selectedRow = self.ui.tableWidge_imageMeta.currentRow()
    if selectedRow == -1:
      return
    selectedImageId = self.ui.tableWidge_imageMeta.item(selectedRow, 0).text()
    
    if(selectedImageId not in self.idToimageData):
      logging.info("{}: Selected image id [id = {}] was not found".format(self.getCurrentTime(), selectedImageId))
      return
    self.currentImageData = self.idToimageData[selectedImageId]
    self.loadNextImage(self.currentImageData)

  def removeAllWhiteSpaces(self, strChain) -> str:
    '''
    removes white spaces within string
    '''
    pattern = r"\s+"
    return re.sub(pattern, "", strChain)

  def getIdsFromString(self, idStr : str) -> List[str]:
    '''
    parses string which contains comma seperated ids
    Parameters:
        idStr (str): string which contains comma seperated ids
    Returns:
        list: contains ids
    '''
    cleanedStr = self.removeAllWhiteSpaces(idStr)
    idsList = cleanedStr.split(",")
    return list(dict.fromkeys(idsList))# remove all duplicates


#Section: Dicom stream
  # Button: Approve
  def approveSegmentation(self):
    statusApproved = self.ui.btn_approved.isChecked()
    statusFlagged = self.ui.btn_mark_revision.isChecked()

    if(statusFlagged or self.finalStatus==self.STATUS.FLAGGED):
      self.ui.btn_mark_revision.setChecked(False)
      self.ui.btn_mark_revision.setDown(False)
    if(statusApproved):
      self.finalStatus = self.STATUS.APPROVED
      self.ui.btn_approved.setChecked(True)
    else:
      self.finalStatus = ""
    self.updateDisplayImageMetaData()

  # Button: Flagge
  def flagSegmentation(self):
    statusApproved = self.ui.btn_approved.isChecked()
    statusFlagged = self.ui.btn_mark_revision.isChecked()

    if(statusApproved or self.finalStatus==self.STATUS.APPROVED):
      self.ui.btn_approved.setChecked(False)
      self.ui.btn_approved.setDown(False)
    if(statusFlagged):
      self.finalStatus = self.STATUS.FLAGGED
      self.ui.btn_mark_revision.setChecked(True)
    else:
      self.finalStatus = ""
    self.updateDisplayImageMetaData()

  # Button: Clear  
  def clearButtons(self):
    self.ui.btn_mark_revision.setChecked(False)
    self.ui.btn_approved.setChecked(False)

    self.ui.btn_mark_revision.setDown(False)
    self.ui.btn_approved.setDown(False)

    self.ui.btn_easy.setChecked(False)
    self.ui.btn_medium.setChecked(False)
    self.ui.btn_hard.setChecked(False)

    self.ui.btn_easy.setDown(False)
    self.ui.btn_medium.setDown(False)
    self.ui.btn_hard.setDown(False)
  
  def disableButtons(self):
    self.ui.btn_easy.setDown(False)
    self.ui.btn_medium.setDown(False)
    self.ui.btn_hard.setDown(False)

  # Button: Easy  
  def setEasy(self):
    levelEasy = self.ui.btn_easy.isChecked()
    levelMedium = self.ui.btn_medium.isChecked()
    levelHard = self.ui.btn_hard.isChecked()

    if(levelEasy):
      self.finalLevel = self.LEVEL.EASY
    
    if(levelEasy==False and self.finalLevel==self.LEVEL.EASY):
      self.finalLevel=""

    if(levelMedium or levelHard or (self.finalLevel == self.LEVEL.HARD) or (self.finalLevel == self.LEVEL.MEDIUM)):
     
      self.ui.btn_medium.setChecked(False)
      self.ui.btn_hard.setChecked(False)
      self.ui.btn_medium.setDown(False)
      self.ui.btn_hard.setDown(False)
    
    self.updateDisplayImageMetaData()
   
  # Button: Medium
  def setMedium(self):
    levelEasy = self.ui.btn_easy.isChecked()
    levelMedium = self.ui.btn_medium.isChecked()
    levelHard = self.ui.btn_hard.isChecked()

    if(levelMedium):
      self.finalLevel = self.LEVEL.MEDIUM
    
    if(levelMedium==False and self.finalLevel==self.LEVEL.MEDIUM):
      self.finalLevel=""
    
    if(levelEasy or levelHard 
      or (self.finalLevel == self.LEVEL.HARD) 
      or (self.finalLevel == self.LEVEL.EASY)):

      self.ui.btn_easy.setChecked(False)
      self.ui.btn_hard.setChecked(False)
      self.ui.btn_easy.setDown(False)
      self.ui.btn_hard.setDown(False)

    self.updateDisplayImageMetaData()

  # Button: Hard  
  def setHard(self):
    levelEasy = self.ui.btn_easy.isChecked()
    levelMedium = self.ui.btn_medium.isChecked()
    levelHard = self.ui.btn_hard.isChecked()

    if(levelHard):
      self.finalLevel = self.LEVEL.HARD
    
    if(levelHard==False and self.finalLevel==self.LEVEL.HARD):
      self.finalLevel=""
    
    if(levelEasy or levelMedium 
      or (self.finalLevel == self.LEVEL.MEDIUM) 
      or (self.finalLevel == self.LEVEL.EASY)):
      
      self.ui.btn_easy.setChecked(False)
      self.ui.btn_medium.setChecked(False)
      self.ui.btn_easy.setDown(False)
      self.ui.btn_medium.setDown(False)

    self.updateDisplayImageMetaData()

  # Button: Next
  def getNextSegmentation(self):
    '''
    after triggering next button:
      1. persist meta data in monai server
      2. update progess bar
      3. load next dicom & segmentation
    '''

    #Persist MetaData
    self.persistMetaInMonaiServer()
    
    #Request Next Image
    self.imageCounter += 1
   
    if(self.imageCounter >= len(self.listImageData)):
      message = "{}: End of list has been reached.".format(self.getCurrentTime())
      logging.warning(message)
      self.displayDialogInfo(title="Warning", message=message)
      self.imageCounter = len(self.listImageData) - 1
      return
    self.updateHorizontalSlider()
    self.currentImageData = self.listImageData[self.imageCounter]
    
    #Displays Next Image
    self.loadNextImage(self.currentImageData)

  # Monai Server: Put
  def persistMetaInMonaiServer(self):
    '''
    Sends the updated meta data of dicom and segmentation to monai-server
    Monai-server incorporates that information into datastore.json file
    '''
    updatedMetaJson = self.updateImageData()
    imageId = self.currentImageData.getName()
    logging.info("updatedMetaJson")
    logging.info(updatedMetaJson)
    if(updatedMetaJson == ""):
      return
    self.logic.updateLabeInfo(imageId, updatedMetaJson)

  def updateImageData(self) -> str:
    '''
    update meta data in information box
    Returns:
      jsonStr (str): json dictionary which is transformed string
                     contains updated meta data
    '''
    self.finalComment = self.ui.plainText_comment.toPlainText()

    logging.info("Status:  "+self.finalLevel)
    if(self.currentImageData.isEqualSegmentationMeta(status=self.finalStatus, level=self.finalLevel, approvedBy=self.selectedReviewer, comment=self.finalComment)):
      logging.info("{}: No changes for image (id='{}')".format(self.getCurrentTime(), self.currentImageData.getName()))
      return ""
    
    self.currentImageData.updateSegmentationMeta(status=self.finalStatus, level=self.finalLevel, approvedBy=self.selectedReviewer, comment=self.finalComment)
    jsonStr = self.currentImageData.getMeta()
    
    if(jsonStr == None):
      logging.info("{}: No update for Image (id='{}')".format(self.getCurrentTime(), self.currentImageData.getName()))
      return ""
    logging.info("{}: Successfully updated Image (id='{}')".format(self.getCurrentTime(), self.currentImageData.getName()))
    return jsonStr
    
  # Button: Previouse
  def getPreviousSegmenation(self):
    '''
    Loads the previous dicom and corresponding segmentation
    after useres tiggers Previous-Button
    '''
    self.imageCounter -= 1
    if(self.imageCounter < 0):
      message = "{}: Lower limit of data set has been reached.".format(self.getCurrentTime())
      logging.warning(message)
      self.displayDialogInfo(title="Warning", message=message)
      self.imageCounter = 0
      return
    self.updateHorizontalSlider()
    self.currentImageData  = self.listImageData[self.imageCounter]
    self.loadNextImage(self.currentImageData)

  def clearImageData(self):
    self.ui.lineEdit_image_id.setText("")
    self.ui.lineEdit_status.setText("")
    self.ui.lineEdit_segmentator.setText("")
    self.ui.lineEdit_level.setText("")
    self.ui.lineEdit_date.setText("")
    self.ui.plainText_comment.setPlainText("")

  def displayImageMetaData(self, imageData):
    '''
    Displays meta info of dicom and segmentation in the info box on slicer

    Parameters:
      imageData (ImageData): Contains meta data (of dicom and segmenation)
    '''
    self.clearImageData()
    self.clearButtons()

    self.ui.lineEdit_image_id.setText(imageData.getName())
    self.ui.lineEdit_segmentator.setText(imageData.getClientId())
    self.ui.lineEdit_date.setText(imageData.getTime())

    self.ui.lineEdit_status.setText(imageData.getStatus())
    self.ui.plainText_comment.setPlainText(imageData.getComment())
    
    
    if(imageData.isApproved()):
      self.finalStatus = self.STATUS.APPROVED
      self.ui.btn_approved.setChecked(True)
      self.ui.btn_approved.setDown(True)
    
    if(imageData.isFlagged()):
      self.finalStatus = self.STATUS.FLAGGED
      self.ui.btn_mark_revision.setChecked(True)
      self.ui.btn_mark_revision.setDown(True)

    if(imageData.getLevel() != ""):
      self.finalLevel = imageData.getLevel()
      if(self.finalLevel == self.LEVEL.EASY):
        self.ui.btn_easy.setDown(True)
        self.ui.btn_easy.setChecked(True)
        self.setEasy()
      if(self.finalLevel == self.LEVEL.MEDIUM):
        self.ui.btn_medium.setDown(True)
        self.ui.btn_medium.setChecked(True)
        self.setMedium()
      if(self.finalLevel == self.LEVEL.HARD):
        self.ui.btn_hard.setDown(True)
        self.ui.btn_hard.setChecked(True)
        self.setHard()
      self.ui.lineEdit_level.setText(imageData.getLevel())

 
  def updateDisplayImageMetaData(self):
    '''
    Displays updated level (easy, medium, hard)
    in the info box on slicer
    '''
    self.ui.lineEdit_status.setText(self.finalStatus)
    self.ui.lineEdit_level.setText(self.finalLevel)
     
  def loadNextImage(self, imageData):
      '''
      Loads original Dicom image and Segmentation into slicer window
      Parameters:
        imageData (ImageData): Contains meta data (of dicom and segmenation)
                              which is required for rest request to monai server
                              in order to get dicom and segmenation (.nrrd).
      '''
      slicer.mrmlScene.Clear()
      self.finalStatus = ""
      self.finalLevel = ""
      self.finalComment = ""

      self.clearButtons()
      self.displayImageMetaData(imageData)
      self.logic.loadDicomAndSegmentation(imageData)

      if(imageData.getStatus() != self.STATUS.NOT_SEGMENTED):
        self.displayLabelOfSegmentation()

#Section: Display label
  def addSegmentator(self):
    self.segmentEditorWidget.setMasterVolumeNodeSelectorVisible(False)
    self.segmentEditorWidget.setSegmentationNodeSelectorVisible(False)
    self.segmentEditorWidget.setSwitchToSegmentationsButtonVisible(False)

    self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    self.segmentEditorWidget.unorderedEffectsVisible = False
    self.segmentEditorWidget.setReadOnly(True)
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
    
class MONAILabelReviewerTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_MONAILabelReviewer1()

  def test_MONAILabelReviewer1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
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
