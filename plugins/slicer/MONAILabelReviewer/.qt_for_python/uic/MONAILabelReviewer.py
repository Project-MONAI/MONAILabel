# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MONAILabelReviewer.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from ctkCollapsibleButton import ctkCollapsibleButton
from qMRMLWidget import qMRMLWidget


class Ui_MONAILabelReviewer(object):
    def setupUi(self, MONAILabelReviewer):
        if not MONAILabelReviewer.objectName():
            MONAILabelReviewer.setObjectName(u"MONAILabelReviewer")
        MONAILabelReviewer.resize(517, 752)
        self.gridLayout = QGridLayout(MONAILabelReviewer)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.btn_basic_mode = QPushButton(MONAILabelReviewer)
        self.btn_basic_mode.setObjectName(u"btn_basic_mode")
        self.btn_basic_mode.setStyleSheet(u"background-color: rgb(118, 214, 255);")
        self.btn_basic_mode.setCheckable(True)
        self.btn_basic_mode.setChecked(True)

        self.horizontalLayout_9.addWidget(self.btn_basic_mode)

        self.btn_reviewers_mode = QPushButton(MONAILabelReviewer)
        self.btn_reviewers_mode.setObjectName(u"btn_reviewers_mode")
        self.btn_reviewers_mode.setStyleSheet(u"background-color: rgb(255, 126, 121);")
        self.btn_reviewers_mode.setCheckable(True)

        self.horizontalLayout_9.addWidget(self.btn_reviewers_mode)


        self.gridLayout.addLayout(self.horizontalLayout_9, 2, 0, 1, 1)

        self.collapsibleButton_search_image = ctkCollapsibleButton(MONAILabelReviewer)
        self.collapsibleButton_search_image.setObjectName(u"collapsibleButton_search_image")
        self.collapsibleButton_search_image.setEnabled(False)
        self.collapsibleButton_search_image.setCollapsed(True)
        self.horizontalLayout = QHBoxLayout(self.collapsibleButton_search_image)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_15 = QVBoxLayout()
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.tabWidget = QTabWidget(self.collapsibleButton_search_image)
        self.tabWidget.setObjectName(u"tabWidget")
        self.ById = QWidget()
        self.ById.setObjectName(u"ById")
        self.ById.setMinimumSize(QSize(252, 0))
        self.verticalLayout_4 = QVBoxLayout(self.ById)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.label_18 = QLabel(self.ById)
        self.label_18.setObjectName(u"label_18")

        self.verticalLayout_4.addWidget(self.label_18)

        self.textEdit_search = QTextEdit(self.ById)
        self.textEdit_search.setObjectName(u"textEdit_search")

        self.verticalLayout_4.addWidget(self.textEdit_search)

        self.btn_search = QPushButton(self.ById)
        self.btn_search.setObjectName(u"btn_search")
        self.btn_search.setStyleSheet(u"background-color: rgb(146, 146, 146);")

        self.verticalLayout_4.addWidget(self.btn_search)

        self.tabWidget.addTab(self.ById, "")
        self.tab_8 = QWidget()
        self.tab_8.setObjectName(u"tab_8")
        self.verticalLayout_6 = QVBoxLayout(self.tab_8)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.label = QLabel(self.tab_8)
        self.label.setObjectName(u"label")

        self.verticalLayout_6.addWidget(self.label)

        self.comboBox_search_annotator = QComboBox(self.tab_8)
        self.comboBox_search_annotator.setObjectName(u"comboBox_search_annotator")

        self.verticalLayout_6.addWidget(self.comboBox_search_annotator)

        self.label_3 = QLabel(self.tab_8)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout_6.addWidget(self.label_3)

        self.comboBox_search_reviewer = QComboBox(self.tab_8)
        self.comboBox_search_reviewer.setObjectName(u"comboBox_search_reviewer")

        self.verticalLayout_6.addWidget(self.comboBox_search_reviewer)

        self.checkBox_search_approved = QCheckBox(self.tab_8)
        self.checkBox_search_approved.setObjectName(u"checkBox_search_approved")

        self.verticalLayout_6.addWidget(self.checkBox_search_approved)

        self.checkBox_search_flagged = QCheckBox(self.tab_8)
        self.checkBox_search_flagged.setObjectName(u"checkBox_search_flagged")

        self.verticalLayout_6.addWidget(self.checkBox_search_flagged)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_6.addItem(self.verticalSpacer_2)

        self.btn_search_annotator_reviewer = QPushButton(self.tab_8)
        self.btn_search_annotator_reviewer.setObjectName(u"btn_search_annotator_reviewer")
        self.btn_search_annotator_reviewer.setStyleSheet(u"background-color: rgb(146, 146, 146);")

        self.verticalLayout_6.addWidget(self.btn_search_annotator_reviewer)

        self.tabWidget.addTab(self.tab_8, "")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.verticalLayout_7 = QVBoxLayout(self.tab)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.label_2 = QLabel(self.tab)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_7.addWidget(self.label_2)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer_3)

        self.checkBox_search_easy = QCheckBox(self.tab)
        self.checkBox_search_easy.setObjectName(u"checkBox_search_easy")

        self.verticalLayout_7.addWidget(self.checkBox_search_easy)

        self.checkBox_search_medium = QCheckBox(self.tab)
        self.checkBox_search_medium.setObjectName(u"checkBox_search_medium")

        self.verticalLayout_7.addWidget(self.checkBox_search_medium)

        self.checkBox_search_hard = QCheckBox(self.tab)
        self.checkBox_search_hard.setObjectName(u"checkBox_search_hard")

        self.verticalLayout_7.addWidget(self.checkBox_search_hard)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer_4)

        self.btn_search_level = QPushButton(self.tab)
        self.btn_search_level.setObjectName(u"btn_search_level")
        self.btn_search_level.setStyleSheet(u"background-color: rgb(146, 146, 146);")

        self.verticalLayout_7.addWidget(self.btn_search_level)

        self.tabWidget.addTab(self.tab, "")

        self.verticalLayout_15.addWidget(self.tabWidget)


        self.horizontalLayout.addLayout(self.verticalLayout_15)

        self.verticalLayout_16 = QVBoxLayout()
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.label_search_result = QLabel(self.collapsibleButton_search_image)
        self.label_search_result.setObjectName(u"label_search_result")

        self.verticalLayout_16.addWidget(self.label_search_result)

        self.tableWidge_imageMeta = QTableWidget(self.collapsibleButton_search_image)
        if (self.tableWidge_imageMeta.columnCount() < 3):
            self.tableWidge_imageMeta.setColumnCount(3)
        font = QFont()
        font.setPointSize(10)
        __qtablewidgetitem = QTableWidgetItem()
        __qtablewidgetitem.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem.setFont(font);
        self.tableWidge_imageMeta.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        __qtablewidgetitem1.setTextAlignment(Qt.AlignCenter);
        __qtablewidgetitem1.setFont(font);
        self.tableWidge_imageMeta.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.tableWidge_imageMeta.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        self.tableWidge_imageMeta.setObjectName(u"tableWidge_imageMeta")
        self.tableWidge_imageMeta.setSortingEnabled(True)

        self.verticalLayout_16.addWidget(self.tableWidge_imageMeta)

        self.btn_show_image = QPushButton(self.collapsibleButton_search_image)
        self.btn_show_image.setObjectName(u"btn_show_image")
        self.btn_show_image.setEnabled(False)

        self.verticalLayout_16.addWidget(self.btn_show_image)


        self.horizontalLayout.addLayout(self.verticalLayout_16)


        self.gridLayout.addWidget(self.collapsibleButton_search_image, 8, 0, 1, 1)

        self.CollapsibleButton = ctkCollapsibleButton(MONAILabelReviewer)
        self.CollapsibleButton.setObjectName(u"CollapsibleButton")
        self.gridLayout_2 = QGridLayout(self.CollapsibleButton)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.comboBox_server_url = QComboBox(self.CollapsibleButton)
        self.comboBox_server_url.setObjectName(u"comboBox_server_url")
        self.comboBox_server_url.setEditable(True)

        self.gridLayout_3.addWidget(self.comboBox_server_url, 1, 1, 1, 1)

        self.label_idx_seg_image = QLabel(self.CollapsibleButton)
        self.label_idx_seg_image.setObjectName(u"label_idx_seg_image")
        self.label_idx_seg_image.setAlignment(Qt.AlignCenter)

        self.gridLayout_3.addWidget(self.label_idx_seg_image, 3, 2, 1, 1)

        self.label_idx_appr_image = QLabel(self.CollapsibleButton)
        self.label_idx_appr_image.setObjectName(u"label_idx_appr_image")
        self.label_idx_appr_image.setAlignment(Qt.AlignCenter)

        self.gridLayout_3.addWidget(self.label_idx_appr_image, 4, 2, 1, 1)

        self.btn_connect_monai = QPushButton(self.CollapsibleButton)
        self.btn_connect_monai.setObjectName(u"btn_connect_monai")
        self.btn_connect_monai.setStyleSheet(u"background-color: rgba(0, 144, 81, 1);")

        self.gridLayout_3.addWidget(self.btn_connect_monai, 1, 2, 1, 1)

        self.label_8 = QLabel(self.CollapsibleButton)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_3.addWidget(self.label_8, 3, 0, 1, 1)

        self.progressBar_approved_total = QProgressBar(self.CollapsibleButton)
        self.progressBar_approved_total.setObjectName(u"progressBar_approved_total")
        self.progressBar_approved_total.setStyleSheet(u"selection-background-color: rgb(255, 147, 0);")
        self.progressBar_approved_total.setValue(0)

        self.gridLayout_3.addWidget(self.progressBar_approved_total, 4, 1, 1, 1)

        self.label_17 = QLabel(self.CollapsibleButton)
        self.label_17.setObjectName(u"label_17")

        self.gridLayout_3.addWidget(self.label_17, 4, 0, 1, 1)

        self.progressBar_segmentation = QProgressBar(self.CollapsibleButton)
        self.progressBar_segmentation.setObjectName(u"progressBar_segmentation")
        self.progressBar_segmentation.setValue(0)

        self.gridLayout_3.addWidget(self.progressBar_segmentation, 3, 1, 1, 1)

        self.label_19 = QLabel(self.CollapsibleButton)
        self.label_19.setObjectName(u"label_19")

        self.gridLayout_3.addWidget(self.label_19, 1, 0, 1, 1)

        self.comboBox_reviewers = QComboBox(self.CollapsibleButton)
        self.comboBox_reviewers.setObjectName(u"comboBox_reviewers")
        self.comboBox_reviewers.setEnabled(True)
        self.comboBox_reviewers.setEditable(True)

        self.gridLayout_3.addWidget(self.comboBox_reviewers, 2, 1, 1, 1)

        self.label_20 = QLabel(self.CollapsibleButton)
        self.label_20.setObjectName(u"label_20")

        self.gridLayout_3.addWidget(self.label_20, 2, 0, 1, 1)


        self.gridLayout_2.addLayout(self.gridLayout_3, 0, 0, 1, 1)


        self.gridLayout.addWidget(self.CollapsibleButton, 4, 0, 1, 1)

        self.collapsibleButton_dicom_evaluation = ctkCollapsibleButton(MONAILabelReviewer)
        self.collapsibleButton_dicom_evaluation.setObjectName(u"collapsibleButton_dicom_evaluation")
        self.collapsibleButton_dicom_evaluation.setEnabled(False)
        self.collapsibleButton_dicom_evaluation.setCollapsed(False)
        self.verticalLayout_12 = QVBoxLayout(self.collapsibleButton_dicom_evaluation)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_level_difficulty = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_level_difficulty.setObjectName(u"label_level_difficulty")

        self.verticalLayout_2.addWidget(self.label_level_difficulty)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.btn_easy = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_easy.setObjectName(u"btn_easy")
        self.btn_easy.setStyleSheet(u"background-color: rgb(0, 250, 146);")

        self.horizontalLayout_5.addWidget(self.btn_easy)

        self.btn_medium = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_medium.setObjectName(u"btn_medium")
        self.btn_medium.setStyleSheet(u"background-color: rgba(255, 251, 0, 179);")

        self.horizontalLayout_5.addWidget(self.btn_medium)

        self.btn_hard = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_hard.setObjectName(u"btn_hard")
        self.btn_hard.setStyleSheet(u"background-color: rgba(255, 38, 0, 179);")

        self.horizontalLayout_5.addWidget(self.btn_hard)


        self.verticalLayout_2.addLayout(self.horizontalLayout_5)


        self.verticalLayout_11.addLayout(self.verticalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.btn_previous = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_previous.setObjectName(u"btn_previous")
        self.btn_previous.setStyleSheet(u"background-color: rgb(255, 147, 0);")

        self.horizontalLayout_3.addWidget(self.btn_previous)

        self.btn_next = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_next.setObjectName(u"btn_next")
        self.btn_next.setStyleSheet(u"background-color: rgb(118, 214, 255);")

        self.horizontalLayout_3.addWidget(self.btn_next)

        self.btn_mark_revision = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_mark_revision.setObjectName(u"btn_mark_revision")

        self.horizontalLayout_3.addWidget(self.btn_mark_revision)

        self.btn_approved = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_approved.setObjectName(u"btn_approved")

        self.horizontalLayout_3.addWidget(self.btn_approved)


        self.verticalLayout_11.addLayout(self.horizontalLayout_3)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_idx_image = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_idx_image.setObjectName(u"label_idx_image")

        self.verticalLayout.addWidget(self.label_idx_image)

        self.horizontalSlider_image_idx = QSlider(self.collapsibleButton_dicom_evaluation)
        self.horizontalSlider_image_idx.setObjectName(u"horizontalSlider_image_idx")
        self.horizontalSlider_image_idx.setEnabled(False)
        self.horizontalSlider_image_idx.setOrientation(Qt.Horizontal)

        self.verticalLayout.addWidget(self.horizontalSlider_image_idx)


        self.verticalLayout_11.addLayout(self.verticalLayout)


        self.verticalLayout_12.addLayout(self.verticalLayout_11)

        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.label_version_labels = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_version_labels.setObjectName(u"label_version_labels")

        self.verticalLayout_10.addWidget(self.label_version_labels)

        self.splitter = QSplitter(self.collapsibleButton_dicom_evaluation)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.comboBox_label_version = QComboBox(self.splitter)
        self.comboBox_label_version.setObjectName(u"comboBox_label_version")
        self.splitter.addWidget(self.comboBox_label_version)
        self.btn_edit_label = QPushButton(self.splitter)
        self.btn_edit_label.setObjectName(u"btn_edit_label")
        self.btn_edit_label.setStyleSheet(u"background-color: rgb(0, 150, 255);")
        self.splitter.addWidget(self.btn_edit_label)

        self.verticalLayout_10.addWidget(self.splitter)

        self.splitter_3 = QSplitter(self.collapsibleButton_dicom_evaluation)
        self.splitter_3.setObjectName(u"splitter_3")
        self.splitter_3.setOrientation(Qt.Horizontal)
        self.btn_overwrite_version = QPushButton(self.splitter_3)
        self.btn_overwrite_version.setObjectName(u"btn_overwrite_version")
        self.btn_overwrite_version.setStyleSheet(u"")
        self.splitter_3.addWidget(self.btn_overwrite_version)
        self.btn_save_new_version = QPushButton(self.splitter_3)
        self.btn_save_new_version.setObjectName(u"btn_save_new_version")
        self.btn_save_new_version.setStyleSheet(u"")
        self.splitter_3.addWidget(self.btn_save_new_version)
        self.btn_delete_version = QPushButton(self.splitter_3)
        self.btn_delete_version.setObjectName(u"btn_delete_version")
        self.splitter_3.addWidget(self.btn_delete_version)

        self.verticalLayout_10.addWidget(self.splitter_3)

        self.btn_update_version = QPushButton(self.collapsibleButton_dicom_evaluation)
        self.btn_update_version.setObjectName(u"btn_update_version")
        self.btn_update_version.setStyleSheet(u"background-color: rgb(115, 250, 121);")

        self.verticalLayout_10.addWidget(self.btn_update_version)


        self.verticalLayout_12.addLayout(self.verticalLayout_10)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.verticalLayout_9 = QVBoxLayout()
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.label_12 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_12.setObjectName(u"label_12")

        self.verticalLayout_9.addWidget(self.label_12)

        self.label_13 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_13.setObjectName(u"label_13")

        self.verticalLayout_9.addWidget(self.label_13)

        self.label_15 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_15.setObjectName(u"label_15")

        self.verticalLayout_9.addWidget(self.label_15)

        self.label_16 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_16.setObjectName(u"label_16")

        self.verticalLayout_9.addWidget(self.label_16)

        self.label_14 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_14.setObjectName(u"label_14")

        self.verticalLayout_9.addWidget(self.label_14)

        self.label_5 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_5.setObjectName(u"label_5")

        self.verticalLayout_9.addWidget(self.label_5)

        self.label_11 = QLabel(self.collapsibleButton_dicom_evaluation)
        self.label_11.setObjectName(u"label_11")

        self.verticalLayout_9.addWidget(self.label_11)


        self.horizontalLayout_2.addLayout(self.verticalLayout_9)

        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.lineEdit_image_id = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_image_id.setObjectName(u"lineEdit_image_id")
        self.lineEdit_image_id.setEnabled(False)

        self.verticalLayout_8.addWidget(self.lineEdit_image_id)

        self.lineEdit_segmentator = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_segmentator.setObjectName(u"lineEdit_segmentator")
        self.lineEdit_segmentator.setEnabled(False)

        self.verticalLayout_8.addWidget(self.lineEdit_segmentator)

        self.lineEdit_date = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_date.setObjectName(u"lineEdit_date")
        self.lineEdit_date.setEnabled(False)

        self.verticalLayout_8.addWidget(self.lineEdit_date)

        self.lineEdit_level = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_level.setObjectName(u"lineEdit_level")
        self.lineEdit_level.setEnabled(False)

        self.verticalLayout_8.addWidget(self.lineEdit_level)

        self.lineEdit_status = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_status.setObjectName(u"lineEdit_status")
        self.lineEdit_status.setEnabled(False)

        self.verticalLayout_8.addWidget(self.lineEdit_status)

        self.lineEdit_editor = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_editor.setObjectName(u"lineEdit_editor")

        self.verticalLayout_8.addWidget(self.lineEdit_editor)

        self.lineEdit_editing_date = QLineEdit(self.collapsibleButton_dicom_evaluation)
        self.lineEdit_editing_date.setObjectName(u"lineEdit_editing_date")

        self.verticalLayout_8.addWidget(self.lineEdit_editing_date)


        self.horizontalLayout_2.addLayout(self.verticalLayout_8)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_2)

        self.plainText_comment = QPlainTextEdit(self.collapsibleButton_dicom_evaluation)
        self.plainText_comment.setObjectName(u"plainText_comment")

        self.horizontalLayout_4.addWidget(self.plainText_comment)


        self.verticalLayout_12.addLayout(self.horizontalLayout_4)


        self.gridLayout.addWidget(self.collapsibleButton_dicom_evaluation, 7, 0, 1, 1)

        self.collapsibleButton_dicom_stream = ctkCollapsibleButton(MONAILabelReviewer)
        self.collapsibleButton_dicom_stream.setObjectName(u"collapsibleButton_dicom_stream")
        self.collapsibleButton_dicom_stream.setEnabled(False)
        self.collapsibleButton_dicom_stream.setCollapsed(True)
        self.gridLayout_5 = QGridLayout(self.collapsibleButton_dicom_stream)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.btn_load = QPushButton(self.collapsibleButton_dicom_stream)
        self.btn_load.setObjectName(u"btn_load")

        self.gridLayout_4.addWidget(self.btn_load, 0, 2, 1, 1)

        self.progressBar_approved_client = QProgressBar(self.collapsibleButton_dicom_stream)
        self.progressBar_approved_client.setObjectName(u"progressBar_approved_client")
        self.progressBar_approved_client.setStyleSheet(u"selection-background-color: rgba(255, 147, 0, 209);")
        self.progressBar_approved_client.setValue(0)

        self.gridLayout_4.addWidget(self.progressBar_approved_client, 2, 1, 1, 1)

        self.comboBox_clients = QComboBox(self.collapsibleButton_dicom_stream)
        self.comboBox_clients.setObjectName(u"comboBox_clients")

        self.gridLayout_4.addWidget(self.comboBox_clients, 0, 1, 1, 1)

        self.label_10 = QLabel(self.collapsibleButton_dicom_stream)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_4.addWidget(self.label_10, 2, 0, 1, 1)

        self.label_9 = QLabel(self.collapsibleButton_dicom_stream)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_4.addWidget(self.label_9, 1, 0, 1, 1)

        self.label_7 = QLabel(self.collapsibleButton_dicom_stream)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_4.addWidget(self.label_7, 0, 0, 1, 1)

        self.progressBar_segmented_client = QProgressBar(self.collapsibleButton_dicom_stream)
        self.progressBar_segmented_client.setObjectName(u"progressBar_segmented_client")
        self.progressBar_segmented_client.setStyleSheet(u"selection-background-color: rgba(78, 157, 246, 209);")
        self.progressBar_segmented_client.setValue(0)
        self.progressBar_segmented_client.setTextVisible(True)

        self.gridLayout_4.addWidget(self.progressBar_segmented_client, 1, 1, 1, 1)

        self.label_6 = QLabel(self.collapsibleButton_dicom_stream)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_4.addWidget(self.label_6, 3, 0, 1, 1)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.checkBox_not_segmented = QCheckBox(self.collapsibleButton_dicom_stream)
        self.checkBox_not_segmented.setObjectName(u"checkBox_not_segmented")
        self.checkBox_not_segmented.setEnabled(False)

        self.verticalLayout_3.addWidget(self.checkBox_not_segmented)

        self.checkBox_flagged = QCheckBox(self.collapsibleButton_dicom_stream)
        self.checkBox_flagged.setObjectName(u"checkBox_flagged")

        self.verticalLayout_3.addWidget(self.checkBox_flagged)


        self.horizontalLayout_6.addLayout(self.verticalLayout_3)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.checkBox_segmented = QCheckBox(self.collapsibleButton_dicom_stream)
        self.checkBox_segmented.setObjectName(u"checkBox_segmented")

        self.verticalLayout_5.addWidget(self.checkBox_segmented)

        self.checkBox_approved = QCheckBox(self.collapsibleButton_dicom_stream)
        self.checkBox_approved.setObjectName(u"checkBox_approved")

        self.verticalLayout_5.addWidget(self.checkBox_approved)


        self.horizontalLayout_6.addLayout(self.verticalLayout_5)


        self.gridLayout_4.addLayout(self.horizontalLayout_6, 3, 1, 1, 1)

        self.label_idx_seg_image_client = QLabel(self.collapsibleButton_dicom_stream)
        self.label_idx_seg_image_client.setObjectName(u"label_idx_seg_image_client")
        self.label_idx_seg_image_client.setAlignment(Qt.AlignCenter)

        self.gridLayout_4.addWidget(self.label_idx_seg_image_client, 1, 2, 1, 1)

        self.label_idx_appr_image_client = QLabel(self.collapsibleButton_dicom_stream)
        self.label_idx_appr_image_client.setObjectName(u"label_idx_appr_image_client")
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
        self.btn_basic_mode.setText(QCoreApplication.translate("MONAILabelReviewer", u"Basic mode", None))
        self.btn_reviewers_mode.setText(QCoreApplication.translate("MONAILabelReviewer", u"Reviewer's mode", None))
        self.collapsibleButton_search_image.setText(QCoreApplication.translate("MONAILabelReviewer", u"Search Images", None))
        self.label_18.setText(QCoreApplication.translate("MONAILabelReviewer", u"Image Ids", None))
        self.textEdit_search.setPlaceholderText(QCoreApplication.translate("MONAILabelReviewer", u"imageId_1, imageId2, ...", None))
        self.btn_search.setText(QCoreApplication.translate("MONAILabelReviewer", u"Search", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.ById), QCoreApplication.translate("MONAILabelReviewer", u"Ids", None))
        self.label.setText(QCoreApplication.translate("MONAILabelReviewer", u"Select annotator", None))
        self.label_3.setText(QCoreApplication.translate("MONAILabelReviewer", u"Select reviewer", None))
        self.checkBox_search_approved.setText(QCoreApplication.translate("MONAILabelReviewer", u"approved", None))
        self.checkBox_search_flagged.setText(QCoreApplication.translate("MONAILabelReviewer", u"flagged", None))
        self.btn_search_annotator_reviewer.setText(QCoreApplication.translate("MONAILabelReviewer", u"Search", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_8), QCoreApplication.translate("MONAILabelReviewer", u"Annotator/Reviewer", None))
        self.label_2.setText(QCoreApplication.translate("MONAILabelReviewer", u"Select level of difficulty", None))
        self.checkBox_search_easy.setText(QCoreApplication.translate("MONAILabelReviewer", u"easy", None))
        self.checkBox_search_medium.setText(QCoreApplication.translate("MONAILabelReviewer", u"medium", None))
        self.checkBox_search_hard.setText(QCoreApplication.translate("MONAILabelReviewer", u"hard", None))
        self.btn_search_level.setText(QCoreApplication.translate("MONAILabelReviewer", u"Search", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MONAILabelReviewer", u"Quality", None))
        self.label_search_result.setText(QCoreApplication.translate("MONAILabelReviewer", u"Result:", None))
        ___qtablewidgetitem = self.tableWidge_imageMeta.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MONAILabelReviewer", u"Image Id", None));
        ___qtablewidgetitem1 = self.tableWidge_imageMeta.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MONAILabelReviewer", u"found", None));
        ___qtablewidgetitem2 = self.tableWidge_imageMeta.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("MONAILabelReviewer", u"segmented", None));
        self.btn_show_image.setText(QCoreApplication.translate("MONAILabelReviewer", u"Show", None))
        self.CollapsibleButton.setText(QCoreApplication.translate("MONAILabelReviewer", u"Server", None))
        self.label_idx_seg_image.setText("")
        self.label_idx_appr_image.setText("")
        self.btn_connect_monai.setText(QCoreApplication.translate("MONAILabelReviewer", u"Connect", None))
        self.label_8.setText(QCoreApplication.translate("MONAILabelReviewer", u"Segmented", None))
        self.label_17.setText(QCoreApplication.translate("MONAILabelReviewer", u"Approved", None))
        self.label_19.setText(QCoreApplication.translate("MONAILabelReviewer", u"Server IP:", None))
        self.label_20.setText(QCoreApplication.translate("MONAILabelReviewer", u"Reviewer:", None))
        self.collapsibleButton_dicom_evaluation.setText(QCoreApplication.translate("MONAILabelReviewer", u"Data Evaluation", None))
        self.label_level_difficulty.setText(QCoreApplication.translate("MONAILabelReviewer", u"Level of difficulty", None))
        self.btn_easy.setText(QCoreApplication.translate("MONAILabelReviewer", u"Easy", None))
        self.btn_medium.setText(QCoreApplication.translate("MONAILabelReviewer", u"Medium", None))
        self.btn_hard.setText(QCoreApplication.translate("MONAILabelReviewer", u"Hard", None))
        self.btn_previous.setText(QCoreApplication.translate("MONAILabelReviewer", u"Previous", None))
        self.btn_next.setText(QCoreApplication.translate("MONAILabelReviewer", u"Next", None))
        self.btn_mark_revision.setText(QCoreApplication.translate("MONAILabelReviewer", u"Flag", None))
        self.btn_approved.setText(QCoreApplication.translate("MONAILabelReviewer", u"Approve", None))
        self.label_idx_image.setText(QCoreApplication.translate("MONAILabelReviewer", u"Image: x/y", None))
        self.label_version_labels.setText(QCoreApplication.translate("MONAILabelReviewer", u"Version of labels", None))
        self.btn_edit_label.setText(QCoreApplication.translate("MONAILabelReviewer", u"Start label edit", None))
        self.btn_overwrite_version.setText(QCoreApplication.translate("MONAILabelReviewer", u"Overwrite this version", None))
        self.btn_save_new_version.setText(QCoreApplication.translate("MONAILabelReviewer", u"Save as new version", None))
        self.btn_delete_version.setText(QCoreApplication.translate("MONAILabelReviewer", u"Delete this version", None))
        self.btn_update_version.setText(QCoreApplication.translate("MONAILabelReviewer", u"Confirm", None))
        self.label_12.setText(QCoreApplication.translate("MONAILabelReviewer", u"Image Id: ", None))
        self.label_13.setText(QCoreApplication.translate("MONAILabelReviewer", u"Annotator:", None))
        self.label_15.setText(QCoreApplication.translate("MONAILabelReviewer", u"Annotation Date:", None))
        self.label_16.setText(QCoreApplication.translate("MONAILabelReviewer", u"Difficulty Level:", None))
        self.label_14.setText(QCoreApplication.translate("MONAILabelReviewer", u"Status:", None))
        self.label_5.setText(QCoreApplication.translate("MONAILabelReviewer", u"Editor: ", None))
        self.label_11.setText(QCoreApplication.translate("MONAILabelReviewer", u"Editing Date:", None))
        self.plainText_comment.setPlaceholderText(QCoreApplication.translate("MONAILabelReviewer", u"Add Comment", None))
        self.collapsibleButton_dicom_stream.setText(QCoreApplication.translate("MONAILabelReviewer", u"Data Set Explorer", None))
        self.btn_load.setText(QCoreApplication.translate("MONAILabelReviewer", u"Load", None))
        self.label_10.setText(QCoreApplication.translate("MONAILabelReviewer", u"Approved", None))
        self.label_9.setText(QCoreApplication.translate("MONAILabelReviewer", u"Segmented", None))
        self.label_7.setText(QCoreApplication.translate("MONAILabelReviewer", u"Annotator", None))
        self.label_6.setText(QCoreApplication.translate("MONAILabelReviewer", u"Filter", None))
        self.checkBox_not_segmented.setText(QCoreApplication.translate("MONAILabelReviewer", u"not segmented", None))
        self.checkBox_flagged.setText(QCoreApplication.translate("MONAILabelReviewer", u"flagged", None))
        self.checkBox_segmented.setText(QCoreApplication.translate("MONAILabelReviewer", u"segmented", None))
        self.checkBox_approved.setText(QCoreApplication.translate("MONAILabelReviewer", u"approved", None))
        self.label_idx_seg_image_client.setText("")
        self.label_idx_appr_image_client.setText("")
        pass
    # retranslateUi

