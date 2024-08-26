# Software Name : SIBR_core
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: Apache 2.0
#
# This software is distributed under the Apache 2.0 License;
# see the LICENSE file for more details.
#
# Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

set(SIBR_PROJECT "openxr")
project(sibr_${SIBR_PROJECT})

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

## Specify target rules
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenXR_INCLUDE_DIR} $ENV{NVTX_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}
    OpenXR::openxr_loader
    sibr_view
)

add_definitions(-DSIBR_OPENXR_EXPORTS -DBOOST_ALL_DYN_LINK)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER ${SIBR_FOLDER})

## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB                         ## mean install also MSVC IDE *.pdb file (DEST according to target type)
)
