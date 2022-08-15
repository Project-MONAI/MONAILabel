################################################################################
# Form generated from reading UI file 'MONAILabelReviewer.ui'
#
# Created by: Qt User Interface Compiler version 5.15.2
#
# WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from ctkCollapsibleButton import ctkCollapsibleButton
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MONAILabelReviewer:
    def setupUi(self, MONAILabelReviewer):
        if not MONAILabelReviewer.objectName():
            MONAILabelReviewer.setObjectName("MONAILabelReviewer")
        MONAILabelReviewer.resize(517, 752)
        self.gridLayout = QGridLayout(MONAILabelReviewer)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.btn_basic_mode = QPushButton(MONAILabelReviewer)
        self.btn_basic_mode.setObjectName("btn_basic_mode")
        self.btn_basic_mode.setStyleSheet("background-color: rgb(118, 214, 255);")
        self.btn_basic_mode.setCheckable(True)
        self.btn_basic_mode.setChecked(True)

        self.horizontalLayout_9.addWidget(self.btn_basic_mode)

        self.btn_reviewers_mode = QPushButton(MONAILabelReviewer)
        self.btn_reviewers_mode.setObjectName("btn_reviewers_mode")
        self.btn_reviewers_mode.setStyleSheet("background-color: rgb(255, 126, 121);")
        self.btn_reviewers_mode.setCheckable(True)

        self.horizontalLayout_9.addWidget(self.btn_reviewers_mode)

        self.gridLayout.addLayout(self.horizontalLayout_9, 2, 0, 1, 1)

        self.collapsibleButton_search_image = ctkCollapsibleButton(MONAILabelReviewer)
        self.collapsibleButton_search_image.setObjectName("collapsibleButton_search_image")
        self.collapsibleButton_search_image.setEnabled(False)
        self.collapsibleButton_search_image.setCollapsed(True)
        self.horizontalLayout = QHBoxLayout(self.collapsibleButton_search_image)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_15 = QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.tabWidget = QTabWidget(self.collapsibleButton_search_image)
        self.tabWidget.setObjectName("tabWidget")
        self.ById = QWidget()
        self.ById.setObjectName("ById")
        self.ById.setMinimumSize(QSize(252, 0))
        self.verticalLayout_4 = QVBoxLayout(self.ById)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_18 = QLabel(self.ById)
        self.label_18.setObjectName("label_18")

        self.verticalLayout_4.addWidget(self.label_18)

        self.textEdit_search = QTextEdit(self.ById)
        self.textEdit_search.setObjectName("textEdit_search")

        self.verticalLayout_4.addWidget(self.textEdit_search)

        self.btn_search = QPushButton(self.ById)
        self.btn_search.setObjectName("btn_search")
        self.btn_search.setStyleSheet("background-color: rgb(146, 146, 146);")

        self.verticalLayout_4.addWidget(self.btn_search)

        self.tabWidget.addTab(self.ById, "")
        self.tab_8 = QWidget()
        self.tab_8.setObjectName("tab_8")
        self.verticalLayout_6 = QVBoxLayout(self.tab_8)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label = QLabel(self.tab_8)
        self.label.setObjectName("label")

        self.verticalLayout_6.addWidget(self.label)

        self.comboBox_search_annotator = QComboBox(self.tab_8)
        self.comboBox_search_annotator.setObjectName("comboBox_search_annotator")

        self.verticalLayout_6.addWidget(self.comboBox_search_annotator)

        self.label_3 = QLabel(self.tab_8)
        self.label_3.setObjectName("label_3")

        self.verticalLayout_6.addWidget(self.label_3)

        self.comboBox_search_reviewer = QComboBox(self.tab_8)
        self.comboBox_search_reviewer.setObjectName("comboBox_search_reviewer")

        self.verticalLayout_6.addWidget(self.comboBox_search_reviewer)

        self.checkBox_search_approved = QCheckBox(self.tab_8)
        self.checkBox_search_approved.setObjectName("checkBox_search_approved")

        self.verticalLayout_6.addWidget(self.checkBox_search_approved)

        self.checkBox_search_flagged = QCheckBox(self.tab_8)
        self.checkBox_search_flagged.setObjectName("checkBox_search_flagged")

        self.verticalLayout_6.addWidget(self.checkBox_search_flagged)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_6.addItem(self.verticalSpacer_2)

        self.btn_search_annotator_reviewer = QPushButton(self.tab_8)
        self.btn_search_annotator_reviewer.setObjectName("btn_search_annotator_reviewer")
        self.btn_search_annotator_reviewer.setStyleSheet("background-color: rgb(146, 146, 146);")

        self.verticalLayout_6.addWidget(self.btn_search_annotator_reviewer)

        self.tabWidget.addTab(self.tab_8, "")
        self.tab = QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_7 = QVBoxLayout(self.tab)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_2 = QLabel(self.tab)
        self.label_2.setObjectName("label_2")

        self.verticalLayout_7.addWidget(self.label_2)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer_3)

        self.checkBox_search_easy = QCheckBox(self.tab)
        self.checkBox_search_easy.setObjectName("checkBox_search_easy")

        self.verticalLayout_7.addWidget(self.checkBox_search_easy)

        self.checkBox_search_medium = QCheckBox(self.tab)
        self.checkBox_search_medium.setObjectName("checkBox_search_medium")

        self.verticalLayout_7.addWidget(self.checkBox_search_medium)

        self.checkBox_search_hard = QCheckBox(self.tab)
        self.checkBox_search_hard.setObjectName("checkBox_search_hard")

        self.verticalLayout_7.addWidget(self.checkBox_search_hard)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer_4)

        self.btn_search_level = QPushButton(self.tab)
        self.btn_search_level.setObjectName("btn_search_level")
        self.btn_search_level.setStyleSheet("background-color: rgb(146, 146, 146);")

        self.verticalLayout_7.addWidget(self.btn_search_level)

        self.tabWidget.addTab(self.tab, "")

        self.verticalLayout_15.addWidget(self.tabWidget)

        self.horizontalLayout.addLayout(self.verticalLayout_15)

        self.verticalLayout_16 = QVBoxLayout()
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.label_search_result = QLabel(self.collapsibleButton_search_image)
        self.label_search_result.setObjectName("label_search_result")

        self.verticalLayout_16.addWidget(self.label_search_result)

        self.tableWidge_imageMeta = QTableWidget(self.collapsibleButton_search_image)
        if self.tableWidge_imageMeta.columnCount() < 3:
            self.tableWidge_imageMeta.setColumnCount(3)
        font = QFont()
        font.setPointSize(10)
        __qtablewidgetitem = QTableWidgetItem()
        __qtablewidgetitem.setTextAlignment(Qt.AlignCenter)
        __qtablewidgetitem.setFont(font)
        self.tableWidge_imageMeta.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        __qtablewidgetitem1.setTextAlignment(Qt.AlignCenter)
        __qtablewidgetitem1.setFont(font)
        self.tableWidge_imageMeta.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.tableWidge_imageMeta.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        self.tableWidge_imageMeta.setObjectName("tableWidge_imageMeta")
        self.tableWidge_imageMeta.setSortingEnabled(True)

        self.verticalLayout_16.addWidget(self.tableWidge_imageMeta)

        self.btn_show_image = QPushButton(self.collapsibleButton_search_image)
        self.btn_show_image.setObjectName("btn_show_image")
        self.btn_show_image.setEnabled(False)

        self.verticalLayout_16.addWidget(self.btn_show_image)

        self.horizontalLayout.addLayout(self.verticalLayout_16)

        self.gridLayout.addWidget(self.collapsibleButton_search_image, 8, 0, 1, 1)

        self.CollapsibleButton = ctkCollapsibleButton(MONAILabelReviewer)
        self.CollapsibleButton.setObjectName("CollapsibleButton")
        self.gridLayout_2 = QGridLayout(self.CollapsibleButton)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.comboBox_server_url = QComboBox(self.CollapsibleButton)
        self.comboBox_server_url.setObjectName("comboBox_server_url")
        self.comboBox_server_url.setEditable(True)

        self.gridLayout_3.addWidget(self.comboBox_server_url, 1, 1, 1, 1)

        self.label_idx_seg_image = QLabel(self.CollapsibleButton)
        self.label_idx_seg_image.setObjectName("label_idx_seg_image")
        self.label_idx_seg_image.setAlignment(Qt.AlignCenter)

        self.gridLayout_3.addWidget(self.label_idx_seg_image, 3, 2, 1, 1)

        self.label_idx_appr_image = QLabel(self.CollapsibleButton)
        self.label_idx_appr_image.setObjectName("label_idx_appr_image")
        self.label_idx_appr_image.setAlignment(Qt.AlignCenter)

        self.gridLayout_3.addWidget(self.label_idx_appr_image, 4, 2, 1, 1)

        self.btn_connect_monai = QPushButton(self.CollapsibleButton)
        self.btn_connect_monai.setObjectName("btn_connect_monai")
        self.btn_connect_monai.setStyleSheet("background-color: rgba(0, 144, 81, 1);")

        self.gridLayout_3.addWidget(self.btn_connect_monai, 1, 2, 1, 1)

        self.label_8 = QLabel(self.CollapsibleButton)
        self.label_8.setObjectName("label_8")

        self.gridLayout_3.addWidget(self.label_8, 3, 0, 1, 1)

        self.progressBar_approved_total = QProgressBar(self.CollapsibleButton)
        self.progressBar_approved_total.setObjectName("progressBar_approved_total")
        self.progressBar_approved_total.setStyleSheet("selection-background-color: rgb(255, 147, 0);")
        self.progressBar_approved_total.setValue(0)

        self.gridLayout_3.addWidget(self.progressBar_approved_total, 4, 1, 1, 1)

        self.label_17 = QLabel(self.CollapsibleButton)
        self.label_17.setObjectName("label_17")

        self.gridLayout_3.addWidget(self.label_17, 4, 0, 1, 1)

        self.progressBar_segmentation = QProgressBar(self.CollapsibleButton)
        self.progressBar_segmentation.setObjectName("progressBar_segmentation")
        self.progressBar_segmentation.setValue(0)

        self.gridLayout_3.addWidget(self.progressBar_segmentation, 3, 1, 1, 1)

        self.label_19 = QLabel(self.CollapsibleButton)
        self.label_19.setObjectName("label_19")

        self.gridLayout_3.addWidget(self.label_19, 1, 0, 1, 1)

        self.comboBox_reviewers = QComboBox(self.CollapsibleButton)
        self.comboBox_reviewers.setObjectName("comboBox_reviewers")
        self.comboBox_reviewers.setEnabled(True)
        self.comboBox_reviewers.setEditable(True)

        self.gridLayout_3.addWidget(self.comboBox_reviewers, 2, 1, 1, 1)

        self.label_20 = QLabel(self.CollapsibleButton)
        self.label_20.setObjectName("label_20")

        self.gridLayout_3.addWidget(self.label_20, 2, 0, 1, 1)

        self.gridLayout_2.addLayout(self.gridLayout_3, 0, 0, 1, 1)

        self.gridLayout.addWidget(self.CollapsibleButton, 4, 0, 1, 1)

        self.collapsibleButton_dicom_evaluation = ctkCollapsibleButton(MONAILabelReviewer)
        self.collapsibleButton_dicom_evaluation.setObjectName("collapsibleButton_dicom_evaluation")
        self.collapsibleButton_dicom_evaluation.setEnabled(False)
        self.collapsibleButton_dicom_evaluation.setCollapsed(False)
        self.verticalLayout_12 = QVBoxLayout(self.collapsibleButton_dicom_evaluation)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_level_difficulty = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_level_difficulty.setObjectName("label_level_difficulty")

        self.verticalLayout_2.addWidget(self.label_level_difficulty)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.btn_easy = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_easy.setObjectName("btn_easy")
        self.btn_easy.setStyleSheet("background-color: rgb(0, 250, 146);")

        self.horizontalLayout_5.addWidget(self.btn_easy)

        self.btn_medium = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_medium.setObjectName("btn_medium")
        self.btn_medium.setStyleSheet("background-color: rgba(255, 251, 0, 179);")

        self.horizontalLayout_5.addWidget(self.btn_medium)

        self.btn_hard = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_hard.setObjectName("btn_hard")
        self.btn_hard.setStyleSheet("background-color: rgba(255, 38, 0, 179);")

        self.horizontalLayout_5.addWidget(self.btn_hard)

        self.verticalLayout_2.addLayout(self.horizontalLayout_5)

        self.verticalLayout_11.addLayout(self.verticalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.btn_previous = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_previous.setObjectName("btn_previous")
        self.btn_previous.setStyleSheet("background-color: rgb(255, 147, 0);")

        self.horizontalLayout_3.addWidget(self.btn_previous)

        self.btn_next = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_next.setObjectName("btn_next")
        self.btn_next.setStyleSheet("background-color: rgb(118, 214, 255);")

        self.horizontalLayout_3.addWidget(self.btn_next)

        self.btn_mark_revision = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_mark_revision.setObjectName("btn_mark_revision")

        self.horizontalLayout_3.addWidget(self.btn_mark_revision)

        self.btn_approved = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_approved.setObjectName("btn_approved")

        self.horizontalLayout_3.addWidget(self.btn_approved)

        self.verticalLayout_11.addLayout(self.horizontalLayout_3)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_idx_image = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_idx_image.setObjectName("label_idx_image")

        self.verticalLayout.addWidget(self.label_idx_image)

        self.horizontalSlider_image_idx = QSlider(self.collapsibleButton_dicom_evaluation)
        self.horizontalSlider_image_idx.setObjectName("horizontalSlider_image_idx")
        self.horizontalSlider_image_idx.setEnabled(False)
        self.horizontalSlider_image_idx.setOrientation(Qt.Horizontal)

        self.verticalLayout.addWidget(self.horizontalSlider_image_idx)

        self.verticalLayout_11.addLayout(self.verticalLayout)

        self.verticalLayout_12.addLayout(self.verticalLayout_11)

        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_version_labels = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_version_labels.setObjectName("label_version_labels")

        self.verticalLayout_10.addWidget(self.label_version_labels)

        self.splitter = QSplitter(self.collapsibleButton_dicom_evaluation)
        self.splitter.setObjectName("splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.comboBox_label_version = QComboBox(self.splitter)
        self.comboBox_label_version.setObjectName("comboBox_label_version")
        self.splitter.addWidget(self.comboBox_label_version)
        self.btn_edit_label = QPushButton(self.splitter)
        self.btn_edit_label.setObjectName("btn_edit_label")
        self.btn_edit_label.setStyleSheet("background-color: rgb(0, 150, 255);")
        self.splitter.addWidget(self.btn_edit_label)

        self.verticalLayout_10.addWidget(self.splitter)

        self.splitter_3 = QSplitter(self.collapsibleButton_dicom_evaluation)
        self.splitter_3.setObjectName("splitter_3")
        self.splitter_3.setOrientation(Qt.Horizontal)
        self.btn_overwrite_version = QPushButton(self.splitter_3)
        self.btn_overwrite_version.setObjectName("btn_overwrite_version")
        self.btn_overwrite_version.setStyleSheet("")
        self.splitter_3.addWidget(self.btn_overwrite_version)
        self.btn_save_new_version = QPushButton(self.splitter_3)
        self.btn_save_new_version.setObjectName("btn_save_new_version")
        self.btn_save_new_version.setStyleSheet("")
        self.splitter_3.addWidget(self.btn_save_new_version)
        self.btn_delete_version = QPushButton(self.splitter_3)
        self.btn_delete_version.setObjectName("btn_delete_version")
        self.splitter_3.addWidget(self.btn_delete_version)

        self.verticalLayout_10.addWidget(self.splitter_3)

        self.btn_update_version = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_update_version.setObjectName("btn_update_version")
        self.btn_update_version.setStyleSheet("background-color: rgb(115, 250, 121);")

        self.verticalLayout_10.addWidget(self.btn_update_version)

        self.verticalLayout_12.addLayout(self.verticalLayout_10)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_9 = QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_12 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_12.setObjectName("label_12")

        self.verticalLayout_9.addWidget(self.label_12)

        self.label_13 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_13.setObjectName("label_13")

        self.verticalLayout_9.addWidget(self.label_13)

        self.label_15 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_15.setObjectName("label_15")

        self.verticalLayout_9.addWidget(self.label_15)

        self.label_16 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_16.setObjectName("label_16")

        self.verticalLayout_9.addWidget(self.label_16)

        self.label_14 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_14.setObjectName("label_14")

        self.verticalLayout_9.addWidget(self.label_14)

        self.label_5 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_5.setObjectName("label_5")

        self.verticalLayout_9.addWidget(self.label_5)

        self.label_11 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_11.setObjectName("label_11")

        self.verticalLayout_9.addWidget(self.label_11)

        self.horizontalLayout_2.addLayout(self.verticalLayout_9)

        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.lineEdit_image_id = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_image_id.setObjectName("lineEdit_image_id")
        self.lineEdit_image_id.setEnabled(False)

        self.verticalLayout_8.addWidget(self.lineEdit_image_id)

        self.lineEdit_segmentator = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_segmentator.setObjectName("lineEdit_segmentator")
        self.lineEdit_segmentator.setEnabled(False)

        self.verticalLayout_8.addWidget(self.lineEdit_segmentator)

        self.lineEdit_date = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_date.setObjectName("lineEdit_date")
        self.lineEdit_date.setEnabled(False)

        self.verticalLayout_8.addWidget(self.lineEdit_date)

        self.lineEdit_level = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_level.setObjectName("lineEdit_level")
        self.lineEdit_level.setEnabled(False)

        self.verticalLayout_8.addWidget(self.lineEdit_level)

        self.lineEdit_status = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_status.setObjectName("lineEdit_status")
        self.lineEdit_status.setEnabled(False)

        self.verticalLayout_8.addWidget(self.lineEdit_status)

        self.lineEdit_editor = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_editor.setObjectName("lineEdit_editor")

        self.verticalLayout_8.addWidget(self.lineEdit_editor)

        self.lineEdit_editing_date = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_editing_date.setObjectName("lineEdit_editing_date")

        self.verticalLayout_8.addWidget(self.lineEdit_editing_date)

        self.horizontalLayout_2.addLayout(self.verticalLayout_8)

        self.horizontalLayout_4.addLayout(self.horizontalLayout_2)

        self.plainText_comment = QPlainTextEdit(self.collapsibleButton_dicom_evaluation)
        self.plainText_comment.setObjectName("plainText_comment")

        self.horizontalLayout_4.addWidget(self.plainText_comment)

        self.verticalLayout_12.addLayout(self.horizontalLayout_4)

        self.gridLayout.addWidget(self.collapsibleButton_dicom_evaluation, 7, 0, 1, 1)

        self.collapsibleButton_dicom_stream = ctkCollapsibleButton(MONAILabelReviewer)
        self.collapsibleButton_dicom_stream.setObjectName("collapsibleButton_dicom_stream")
        self.collapsibleButton_dicom_stream.setEnabled(False)
        self.collapsibleButton_dicom_stream.setCollapsed(True)
        self.gridLayout_5 = QGridLayout(self.collapsibleButton_dicom_stream)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.btn_load = QPushButton(self.collapsibleButton_dicom_stream)
        self.btn_load.setObjectName("btn_load")

        self.gridLayout_4.addWidget(self.btn_load, 0, 2, 1, 1)

        self.progressBar_approved_client = QProgressBar(self.collapsibleButton_dicom_stream)
        self.progressBar_approved_client.setObjectName("progressBar_approved_client")
        self.progressBar_approved_client.setStyleSheet("selection-background-color: rgba(255, 147, 0, 209);")
        self.progressBar_approved_client.setValue(0)

        self.gridLayout_4.addWidget(self.progressBar_approved_client, 2, 1, 1, 1)

        self.comboBox_clients = QComboBox(self.collapsibleButton_dicom_stream)
        self.comboBox_clients.setObjectName("comboBox_clients")

        self.gridLayout_4.addWidget(self.comboBox_clients, 0, 1, 1, 1)

        self.label_10 = QLabel(self.collapsibleButton_dicom_stream)
        self.label_10.setObjectName("label_10")

        self.gridLayout_4.addWidget(self.label_10, 2, 0, 1, 1)

        self.label_9 = QLabel(self.collapsibleButton_dicom_stream)
        self.label_9.setObjectName("label_9")

        self.gridLayout_4.addWidget(self.label_9, 1, 0, 1, 1)

        self.label_7 = QLabel(self.collapsibleButton_dicom_stream)
        self.label_7.setObjectName("label_7")

        self.gridLayout_4.addWidget(self.label_7, 0, 0, 1, 1)

        self.progressBar_segmented_client = QProgressBar(self.collapsibleButton_dicom_stream)
        self.progressBar_segmented_client.setObjectName("progressBar_segmented_client")
        self.progressBar_segmented_client.setStyleSheet("selection-background-color: rgba(78, 157, 246, 209);")
        self.progressBar_segmented_client.setValue(0)
        self.progressBar_segmented_client.setTextVisible(True)

        self.gridLayout_4.addWidget(self.progressBar_segmented_client, 1, 1, 1, 1)

        self.label_6 = QLabel(self.collapsibleButton_dicom_stream)
        self.label_6.setObjectName("label_6")

        self.gridLayout_4.addWidget(self.label_6, 3, 0, 1, 1)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.checkBox_not_segmented = QCheckBox(self.collapsibleButton_dicom_stream)
        self.checkBox_not_segmented.setObjectName("checkBox_not_segmented")
        self.checkBox_not_segmented.setEnabled(False)

        self.verticalLayout_3.addWidget(self.checkBox_not_segmented)

        self.checkBox_flagged = QCheckBox(self.collapsibleButton_dicom_stream)
        self.checkBox_flagged.setObjectName("checkBox_flagged")

        self.verticalLayout_3.addWidget(self.checkBox_flagged)

        self.horizontalLayout_6.addLayout(self.verticalLayout_3)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.checkBox_segmented = QCheckBox(self.collapsibleButton_dicom_stream)
        self.checkBox_segmented.setObjectName("checkBox_segmented")

        self.verticalLayout_5.addWidget(self.checkBox_segmented)

        self.checkBox_approved = QCheckBox(self.collapsibleButton_dicom_stream)
        self.checkBox_approved.setObjectName("checkBox_approved")

        self.verticalLayout_5.addWidget(self.checkBox_approved)

        self.horizontalLayout_6.addLayout(self.verticalLayout_5)

        self.gridLayout_4.addLayout(self.horizontalLayout_6, 3, 1, 1, 1)

        self.label_idx_seg_image_client = QLabel(self.collapsibleButton_dicom_stream)
        self.label_idx_seg_image_client.setObjectName("label_idx_seg_image_client")
        self.label_idx_seg_image_client.setAlignment(Qt.AlignCenter)

        self.gridLayout_4.addWidget(self.label_idx_seg_image_client, 1, 2, 1, 1)

        self.label_idx_appr_image_client = QLabel(self.collapsibleButton_dicom_stream)
        self.label_idx_appr_image_client.setObjectName("label_idx_appr_image_client")
        self.label_idx_appr_image_client.setAlignment(Qt.AlignCenter)

        self.gridLayout_4.addWidget(self.label_idx_appr_image_client, 2, 2, 1, 1)

        self.gridLayout_5.addLayout(self.gridLayout_4, 0, 0, 1, 1)

        self.gridLayout.addWidget(self.collapsibleButton_dicom_stream, 6, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 9, 0, 1, 1)

        self.retranslateUi(MONAILabelReviewer)

        self.tabWidget.setCurrentIndex(2)

        QMetaObject.connectSlotsByName(MONAILabelReviewer)

    # setupUi

    def retranslateUi(self, MONAILabelReviewer):
        self.btn_basic_mode.setText(QCoreApplication.translate("MONAILabelReviewer", "Basic mode", None))
        self.btn_reviewers_mode.setText(QCoreApplication.translate("MONAILabelReviewer", "Reviewer's mode", None))
        self.collapsibleButton_search_image.setText(
            QCoreApplication.translate("MONAILabelReviewer", "Search Images", None)
        )
        self.label_18.setText(QCoreApplication.translate("MONAILabelReviewer", "Image Ids", None))
        self.textEdit_search.setPlaceholderText(
            QCoreApplication.translate("MONAILabelReviewer", "imageId_1, imageId2, ...", None)
        )
        self.btn_search.setText(QCoreApplication.translate("MONAILabelReviewer", "Search", None))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.ById), QCoreApplication.translate("MONAILabelReviewer", "Ids", None)
        )
        self.label.setText(QCoreApplication.translate("MONAILabelReviewer", "Select annotator", None))
        self.label_3.setText(QCoreApplication.translate("MONAILabelReviewer", "Select reviewer", None))
        self.checkBox_search_approved.setText(QCoreApplication.translate("MONAILabelReviewer", "approved", None))
        self.checkBox_search_flagged.setText(QCoreApplication.translate("MONAILabelReviewer", "flagged", None))
        self.btn_search_annotator_reviewer.setText(QCoreApplication.translate("MONAILabelReviewer", "Search", None))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_8),
            QCoreApplication.translate("MONAILabelReviewer", "Annotator/Reviewer", None),
        )
        self.label_2.setText(QCoreApplication.translate("MONAILabelReviewer", "Select level of difficulty", None))
        self.checkBox_search_easy.setText(QCoreApplication.translate("MONAILabelReviewer", "easy", None))
        self.checkBox_search_medium.setText(QCoreApplication.translate("MONAILabelReviewer", "medium", None))
        self.checkBox_search_hard.setText(QCoreApplication.translate("MONAILabelReviewer", "hard", None))
        self.btn_search_level.setText(QCoreApplication.translate("MONAILabelReviewer", "Search", None))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MONAILabelReviewer", "Quality", None)
        )
        self.label_search_result.setText(QCoreApplication.translate("MONAILabelReviewer", "Result:", None))
        ___qtablewidgetitem = self.tableWidge_imageMeta.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MONAILabelReviewer", "Image Id", None))
        ___qtablewidgetitem1 = self.tableWidge_imageMeta.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MONAILabelReviewer", "found", None))
        ___qtablewidgetitem2 = self.tableWidge_imageMeta.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("MONAILabelReviewer", "segmented", None))
        self.btn_show_image.setText(QCoreApplication.translate("MONAILabelReviewer", "Show", None))
        self.CollapsibleButton.setText(QCoreApplication.translate("MONAILabelReviewer", "Server", None))
        self.label_idx_seg_image.setText("")
        self.label_idx_appr_image.setText("")
        self.btn_connect_monai.setText(QCoreApplication.translate("MONAILabelReviewer", "Connect", None))
        self.label_8.setText(QCoreApplication.translate("MONAILabelReviewer", "Segmented", None))
        self.label_17.setText(QCoreApplication.translate("MONAILabelReviewer", "Approved", None))
        self.label_19.setText(QCoreApplication.translate("MONAILabelReviewer", "Server IP:", None))
        self.label_20.setText(QCoreApplication.translate("MONAILabelReviewer", "Reviewer:", None))
        self.collapsibleButton_dicom_evaluation.setText(
            QCoreApplication.translate("MONAILabelReviewer", "Data Evaluation", None)
        )
        self.label_level_difficulty.setText(
            QCoreApplication.translate("MONAILabelReviewer", "Level of difficulty", None)
        )
        self.btn_easy.setText(QCoreApplication.translate("MONAILabelReviewer", "Easy", None))
        self.btn_medium.setText(QCoreApplication.translate("MONAILabelReviewer", "Medium", None))
        self.btn_hard.setText(QCoreApplication.translate("MONAILabelReviewer", "Hard", None))
        self.btn_previous.setText(QCoreApplication.translate("MONAILabelReviewer", "Previous", None))
        self.btn_next.setText(QCoreApplication.translate("MONAILabelReviewer", "Next", None))
        self.btn_mark_revision.setText(QCoreApplication.translate("MONAILabelReviewer", "Flag", None))
        self.btn_approved.setText(QCoreApplication.translate("MONAILabelReviewer", "Approve", None))
        self.label_idx_image.setText(QCoreApplication.translate("MONAILabelReviewer", "Image: x/y", None))
        self.label_version_labels.setText(QCoreApplication.translate("MONAILabelReviewer", "Version of labels", None))
        self.btn_edit_label.setText(QCoreApplication.translate("MONAILabelReviewer", "Start label edit", None))
        self.btn_overwrite_version.setText(
            QCoreApplication.translate("MONAILabelReviewer", "Overwrite this version", None)
        )
        self.btn_save_new_version.setText(QCoreApplication.translate("MONAILabelReviewer", "Save as new version", None))
        self.btn_delete_version.setText(QCoreApplication.translate("MONAILabelReviewer", "Delete this version", None))
        self.btn_update_version.setText(QCoreApplication.translate("MONAILabelReviewer", "Confirm", None))
        self.label_12.setText(QCoreApplication.translate("MONAILabelReviewer", "Image Id: ", None))
        self.label_13.setText(QCoreApplication.translate("MONAILabelReviewer", "Annotator:", None))
        self.label_15.setText(QCoreApplication.translate("MONAILabelReviewer", "Annotation Date:", None))
        self.label_16.setText(QCoreApplication.translate("MONAILabelReviewer", "Difficulty Level:", None))
        self.label_14.setText(QCoreApplication.translate("MONAILabelReviewer", "Status:", None))
        self.label_5.setText(QCoreApplication.translate("MONAILabelReviewer", "Editor: ", None))
        self.label_11.setText(QCoreApplication.translate("MONAILabelReviewer", "Editing Date:", None))
        self.plainText_comment.setPlaceholderText(QCoreApplication.translate("MONAILabelReviewer", "Add Comment", None))
        self.collapsibleButton_dicom_stream.setText(
            QCoreApplication.translate("MONAILabelReviewer", "Data Set Explorer", None)
        )
        self.btn_load.setText(QCoreApplication.translate("MONAILabelReviewer", "Load", None))
        self.label_10.setText(QCoreApplication.translate("MONAILabelReviewer", "Approved", None))
        self.label_9.setText(QCoreApplication.translate("MONAILabelReviewer", "Segmented", None))
        self.label_7.setText(QCoreApplication.translate("MONAILabelReviewer", "Annotator", None))
        self.label_6.setText(QCoreApplication.translate("MONAILabelReviewer", "Filter", None))
        self.checkBox_not_segmented.setText(QCoreApplication.translate("MONAILabelReviewer", "not segmented", None))
        self.checkBox_flagged.setText(QCoreApplication.translate("MONAILabelReviewer", "flagged", None))
        self.checkBox_segmented.setText(QCoreApplication.translate("MONAILabelReviewer", "segmented", None))
        self.checkBox_approved.setText(QCoreApplication.translate("MONAILabelReviewer", "approved", None))
        self.label_idx_seg_image_client.setText("")
        self.label_idx_appr_image_client.setText("")
        pass

    # retranslateUi
