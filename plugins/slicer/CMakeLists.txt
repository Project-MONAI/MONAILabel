cmake_minimum_required(VERSION 3.13.4)

project(MONAI-Label)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/MONAILabel")
set(EXTENSION_CATEGORY "Active Learning")
set(EXTENSION_CONTRIBUTORS "NVIDIA, KCL")
set(EXTENSION_DESCRIPTION "This is Active Learning solution developed under project MONAILabel")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/MONAILabel.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/MONAILabel/Screenshots/1.png")
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
