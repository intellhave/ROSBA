cmake_minimum_required(VERSION 2.6)
project(ROSBA)

set (CMAKE_CXX_STANDARD 11)
set (V3D_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set (V3D_INCLUDE_DIRS ${V3D_DIR}/.)


set (EXTRA_INC_DIRS ${EXTRA_INC_DIRS} 
    /usr/local/include/eigen3/)

include (Config/v3d_macros.cmake)
include (Config/local_config.cmake)

include_directories(${V3D_INCLUDE_DIRS} ${EXTRA_INC_DIRS})

source_group("Base" REGULAR_EXPRESSION Base/.*cpp|Base.*h)
set (BASE_SRC
    Base/v3d_exception.h
    Base/v3d_serialization.h
    Base/v3d_vrmlio.h
    Base/v3d_image.cpp
    )

if (V3DLIB_ENABLE_SOCKETS)
    set (BASE_SRC ${BASE_SRC} Base/v3d_socket.h)

endif (V3DLIB_ENABLE_SOCKETS)

source_group("Math" REGULAR_EXPRESSION Math/.*cpp|Math.*h)
set (MATH_SRC
    Math/v3d_linear.h
    Math/v3d_linearbase.h
    Math/v3d_mathutilities.h
    Math/v3d_nonlinlsq.h
    Math/v3d_nonlinlsq.cpp
    Math/v3d_optimization.h
    Math/v3d_optimization.cpp
    Math/v3d_optimization_lm.cpp
    )

source_group("Geometry" REGULAR_EXPRESSION Geometry/.*cpp|Geometry.*h)

set (GEOMETRY_SRC
    Geometry/v3d_cameramatrix.h
    Geometry/v3d_distortion.h
    Geometry/v3d_metricbundle.h
    Geometry/v3d_metricbundle.cpp
    Geometry/v3d_stereobundle.h
    Geometry/v3d_stereobundle.cpp) 

set (ALL_SRC
    ${BASE_SRC} ${MATH_SRC} ${GEOMETRY_SRC} 
    )


include_directories(${V3D_INCLUDE_DIRS} )
link_directories (${NORMAL_LIBRARY_DIRS} )

set (EIGEN_FLAGS '-D EIGEN_USE_LAPACKE=1 -lm -lblas -llapack -llapacke')


# set (OPTIMIZER schur_irls)
# set (OPTIMIZER schur_lifting)
# set (OPTIMIZER schur_gnc)
# set (OPTIMIZER schur_gncp)
# set (OPTIMIZER schur_filter)
set (OPTIMIZER schur_gemm)
# set (OPTIMIZER schur_gnc_moo)
# set (OPTIMIZER nonrobust)

set (MAIN_FILE bundle_large_robust.cpp)

if (${OPTIMIZER} STREQUAL "schur_irls")
    add_definitions(-DUSE_SCHUR_IRLS)
    set (EXE_NAME schur_irls)
    # uncomment this to use non-robust 
    # add_definitions(-DUSE_LEAST_SQUARES)
elseif(${OPTIMIZER} STREQUAL "schur_lifting")
    add_definitions(-DUSE_SCHUR_LIFTING)
    set (EXE_NAME schur_lifting)
elseif(${OPTIMIZER} STREQUAL "schur_gnc")
    add_definitions(-DUSE_SCHUR_GNC)
    set (EXE_NAME schur_gnc)
elseif(${OPTIMIZER} STREQUAL "schur_gnc_gemm")
    add_definitions(-DUSE_SCHUR_GNC_GEMM)
    set (EXE_NAME schur_gnc_gemm)
elseif(${OPTIMIZER} STREQUAL "schur_gnc_moo")
    add_definitions(-DUSE_SCHUR_GNC_MOO)
    set (EXE_NAME schur_gnc_moo)
elseif(${OPTIMIZER} STREQUAL "schur_filter")
    add_definitions(-DUSE_SCHUR_FILTER)
    set (EXE_NAME schur_filter)
elseif(${OPTIMIZER} STREQUAL "schur_gemm")
    add_definitions(-DUSE_SCHUR_GEMM)
    set (EXE_NAME schur_gemm)
else()
    #Default is IRLS
    add_definitions(-DUSE_IRLS)
    set (EXE_NAME irls)
endif()

set (EXE_NAME bundle_${EXE_NAME})
message(${EXE_NAME})

set (CMAKE_BUILD_TYPE Release)

add_executable (${EXE_NAME} Apps/${MAIN_FILE} ${ALL_SRC})
target_link_libraries(${EXE_NAME} colamd )

if (${CMAKE_BUILD_TYPE} EQUAL Release)
    target_compile_options(${EXE_NAME} PRIVATE -O3 -m64)
endif()

