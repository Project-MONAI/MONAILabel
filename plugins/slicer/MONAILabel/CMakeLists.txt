#-----------------------------------------------------------------------------
set(MODULE_NAME MONAILabel)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/client.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Icons/refresh-icon.png
  Resources/Icons/download.png
  Resources/Icons/save.png
  Resources/Icons/segment.png
  Resources/Icons/stop.png
  Resources/Icons/training.png
  Resources/Icons/upload.svg
  Resources/Icons/contour.svg
  Resources/UI/${MODULE_NAME}.ui
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
