cmake_minimum_required(VERSION 3.22)
project(level_set_method)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(BOOST_ROOT "/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/BOOST" )
set(Boost_NO_BOOST_CMAKE ON CACHE BOOL "" FORCE)
set(CUDA_ON_CPU CACHE BOOL "Make Cuda work on heap")

ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)

# Find MPI
find_package(MPI REQUIRED)
find_package(Boost COMPONENTS filesystem system log log_setup thread context REQUIRED)


link_directories(
		/Users/jstark/openfpm/OpenFPM_install_master/openfpm_devices/lib
		/Users/jstark/openfpm/OpenFPM_install_master/openfpm_pdata/lib
		/Users/jstark/openfpm/OpenFPM_install_master/openfpm_vcluster/lib
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/VCDEVEL/lib
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/METIS/lib
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/PARMETIS/lib
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/BOOST/lib
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/HDF5/lib
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/LIBHILBERT/lib
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/PETSC/lib
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/OPENBLAS/lib
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/SUITESPARSE/lib )


add_executable(
		level_set_method
		main.cpp
)

# Supress warning
target_compile_options(level_set_method PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wno-macro-redefined >)

# Link against MPI
target_link_libraries(level_set_method ${MPI_LIBRARIES} vcluster ofpm_pdata ofpmmemory parmetis metis
		hdf5 libhilbert Vc petsc openblas umfpack amd btf camd ccolamd cholmod colamd cxsparse klu dl rbio spqr
		suitesparseconfig dl boost_iostreams boost_program_options boost_filesystem boost_system boost_filesystem
		boost_context)

target_include_directories(level_set_method PUBLIC /.
		/Users/jstark/openfpm/OpenFPM_install_master/openfpm_numerics/include
		/Users/jstark/openfpm/OpenFPM_install_master/openfpm_pdata/include/config
		/Users/jstark/openfpm/OpenFPM_install_master/openfpm_pdata/include
		/Users/jstark/openfpm/OpenFPM_install_master/openfpm_data/include
		/Users/jstark/openfpm/OpenFPM_install_master/openfpm_vcluster/include
		/Users/jstark/openfpm/OpenFPM_install_master/openfpm_io/include
		/Users/jstark/openfpm/OpenFPM_install_master/openfpm_devices/include
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/VCDEVEL/include
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/METIS/include
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/PARMETIS/include
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/BOOST/include
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/HDF5/include
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/LIBHILBERT/include
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/PETSC/include
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/OPENBLAS/include
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/SUITESPARSE/include
		/Users/jstark/openfpm/OPENFPM_DEPENDENCIES_master/EIGEN)


# Include MPI includes
include_directories(${MPI_INCLUDE_PATH})
if(MPI_COMPILE_FLAGS)
	set_target_properties(level_set_method PROPERTIES
			COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
	set_target_properties(level_set_method PROPERTIES
			LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

