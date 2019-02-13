cmake_minimum_required (VERSION 3.6)
project (UNIPAR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -lpthread")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -lpthread")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH}) # To allow CMake to locate our Find*.cmake files

add_subdirectory(sub)
add_subdirectory(hash)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    # icpc
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
        message(icpc)
        set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        message(g++)
        set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif ()
endif ()

set(SOURCE_FILES 
	./src/comm.c ./src/contig.c ./src/dbgraph.c 
	./src/msp.c ./src/preprocess.c
	./src/main.c)

add_executable(unipar ${SOURCE_FILES})

find_package("CUDA" REQUIRED)	
    macro(cuda_add_compile_options options)
        list(APPEND CUDA_NVCC_FLAGS "${options}")
    endmacro()
    cuda_add_compile_options("-gencode arch=compute_35,code=sm_35 -Xptxas -dlcm=cg")
    cuda_add_compile_options("-std=c++11 -g -G -lpthread")
    cuda_add_compile_options("-Xcompiler ${OpenMP_CXX_FLAGS}")
    
cuda_add_library(culib SHARED 
	./src/comm.cu ./src/contig.cu 
	./src/dbgraph.cu ./src/msp.cu ./src/preprocess.cu)

link_directories(${CMAKE_BINARY_DIR}/hash)
link_directories(${CMAKE_BINARY_DIR}/sub)
target_link_libraries(unipar ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(unipar tbbsort)
target_link_libraries(unipar distribute)
target_link_libraries(unipar sublib)
target_link_libraries(unipar hash)
target_link_libraries(unipar culib)
target_link_libraries(unipar subculib)

set_target_properties(unipar PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

