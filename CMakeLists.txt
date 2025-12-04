
cmake_minimum_required(VERSION 3.15)

project(dem_bones VERSION 1.0 DESCRIPTION "DemBones Python bindings for Maya" LANGUAGES CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 模块路径
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)

# 路径
set(DEMBONES_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/DemBones)
set(EIGEN_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/Eigen)

# 自动搜索源文件
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(GLOB_RECURSE DEMBONES_FILES "${DEMBONES_ROOT_DIR}/include/DemBones/*.h")

# OpenMP
find_package(OpenMP)

# Maya
find_package(Maya REQUIRED)
find_package(MayaPython REQUIRED)

# 添加 pybind11
add_subdirectory(extern/pybind11)

# 创建 Eigen INTERFACE target
add_library(Eigen INTERFACE)
target_include_directories(Eigen INTERFACE ${EIGEN_ROOT_DIR})

# 创建 Python 模块
pybind11_add_module(_core ${SOURCE_FILES} ${DEMBONES_FILES})

# 链接库
target_link_libraries(_core PRIVATE Maya::Maya Eigen pybind11::module)

# 包含目录
target_include_directories(_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${DEMBONES_ROOT_DIR}/include
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# OpenMP 支持
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Windows 特定编译选项
if(WIN32 AND MSVC)
    target_compile_options(_core PRIVATE $<$<CONFIG:DEBUG>:/bigobj>)
endif()

# 输出路径
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
# 安装逻辑
include(GNUInstallDirs)
install(TARGETS _core LIBRARY DESTINATION ".")
