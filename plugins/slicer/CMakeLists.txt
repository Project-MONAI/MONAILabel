cmake_minimum_required(VERSION 3.13.4)

project(MONAILabel)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer/MONAILabel")
set(EXTENSION_CATEGORY "Active Learning")
set(EXTENSION_CONTRIBUTORS "NVIDIA, KCL")
set(EXTENSION_DESCRIPTION "This is Active Learning solution developed under project MONAILabel")
set(EXTENSION_ICONURL "https://github.com/Project-MONAI/MONAILabel/raw/main/plugins/slicer/MONAILabel/Resources/Icons/MONAILabel.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/Project-MONAI/MONAILabel/raw/main/plugins/slicer/MONAILabel/Screenshots/1.png https://github.com/Project-MONAI/MONAILabel/raw/main/plugins/slicer/MONAILabel/Screenshots/2.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MONAILabel)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
