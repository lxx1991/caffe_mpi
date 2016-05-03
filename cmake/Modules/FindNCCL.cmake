# Try to find the NCCL libraries and headers
#  NCCL_FOUND - system has LMDB lib
#  NCCL_INCLUDE_DIR - the LMDB include directory
#  NCCL_LIBRARIES - Libraries needed to use LMDB


find_path(NCCL_INCLUDE_DIR NAMES  nccl.h PATHS "$ENV{NCCL_DIR}/include" ${CUDA_TOOLKIT_INCLUDE})
find_library(NCCL_LIBRARIES NAMES libnccl.so   PATHS "$ENV{NCCL_DIR}/lib" "${CUDA_TOOLKIT_INCLUDE}/../lib64")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARIES )

if(NCCL_FOUND)
    message(STATUS "Found nccl    (include: ${NCCL_INCLUDE_DIR}, library: ${NCCL_LIBRARIES})")
    mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARIES)
endif()
