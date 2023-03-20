################################################################################
# General Information
################################################################################

cmake_minimum_required(VERSION 3.1)
project(baseMeshExtraction)

# Add your project files
# Create an executable
file(GLOB SRCFILES  
    ../../Upsampling/*.h
    ../../Decomposition/*.h
    ../../MeshLib/*.h
    ../../SecondFundamentalForm/*.h
    ../../TFWShell/*.h
    ../../Optimization/*.h
    ../../*.h
    ../../*.hpp
    *.h
    ../../Decomposition/*.cpp
    ../../Upsampling/*.cpp
    ../../MeshLib/*.cpp
    ../../SecondFundamentalForm/*.cpp
    ../../TFWShell/*.cpp
    ../../Optimization/*.cpp
    ../../*.cpp
    *.cpp)
    
add_executable(
        ${PROJECT_NAME}_bin
        ${SRCFILES}
        )

set_target_properties(${PROJECT_NAME}_bin PROPERTIES CXX_STANDARD 17)
set_target_properties(${PROJECT_NAME}_bin PROPERTIES CXX_STANDARD_REQUIRED ON)

# Link settings
target_include_directories(${PROJECT_NAME}_bin PUBLIC ${spectra_SOURCE_DIR}/include/Spectra)
target_link_libraries( ${PROJECT_NAME}_bin 
        polyscope 
        geometry-central
        TBB::tbb 
        tbb_static 
        tbbmalloc_static
        igl::core 
        igl_restricted::triangle 
        igl_copyleft::comiso 
        ${SUITESPARSE_LIBRARIES} ${METIS_LIBRARY}
        CLI11::CLI11)