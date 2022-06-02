#-----------------------------------------------------------------------------
set(MODULE_NAME MONAILabelReviewer)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__
  ${MODULE_NAME}Lib/DataStoreKeys.py
  ${MODULE_NAME}Lib/ImageData.py
  ${MODULE_NAME}Lib/ImageDataController.py
  ${MODULE_NAME}Lib/ImageDataExtractor.py
  ${MODULE_NAME}Lib/JsonParser.py
  ${MODULE_NAME}Lib/MONAILabelReviewerEnum.py
  ${MODULE_NAME}Lib/MonaiServerREST.py
  ${MODULE_NAME}Lib/SegmentationMeta.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/UI/DialogInfo.ui
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
