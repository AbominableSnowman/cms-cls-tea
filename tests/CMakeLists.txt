cmake_minimum_required(VERSION 3.22)
project(myProject)

set(CMAKE_CXX_STANDARD 14)

set(BOOST_ROOT "/home/jarryd/project/openfpm_dependencies_master/BOOST" )
set(Boost_NO_BOOST_CMAKE ON CACHE BOOL "" FORCE)

ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)

# Find MPI
find_package(MPI REQUIRED)
find_package(Boost COMPONENTS filesystem system log log_setup thread REQUIRED)

link_directories(/home/jarryd/project/openfpm_install_master/openfpm_devices/lib
                 /home/jarryd/project/openfpm_install_master/openfpm_pdata/lib
                 /home/jarryd/project/openfpm_install_master/openfpm_vcluster/lib
                 /home/jarryd/project/openfpm_dependencies_master/VCDEVEL/lib
                 /home/jarryd/project/openfpm_dependencies_master/METIS/lib
                 /home/jarryd/project/openfpm_dependencies_master/PARMETIS/lib
                 /home/jarryd/project/openfpm_dependencies_master/BOOST/lib
                 /home/jarryd/project/openfpm_dependencies_master/HDF5/lib
                 /home/jarryd/project/openfpm_dependencies_master/LIBHILBERT/lib
                 /home/jarryd/project/openfpm_dependencies_master/PETSC/lib
                 /home/jarryd/project/openfpm_dependencies_master/OPENBLAS/lib
                 /home/jarryd/project/openfpm_dependencies_master/SUITESPARSE/lib )


add_executable(
                tests
                main.cpp
)

# Supress warning
target_compile_options(tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wno-macro-redefined >)

# Link against MPI
target_link_libraries(tests ${MPI_LIBRARIES} vcluster ofpm_pdata ofpmmemory parmetis metis boost_iostreams boost_program_options hdf5 libhilbert Vc dl boost_filesystem boost_system boost_filesystem boost_system)

target_include_directories(tests PUBLIC /.
                        /home/jarryd/project/openfpm_install_master/openfpm_numerics/include 
                        /home/jarryd/project/openfpm_install_master/openfpm_pdata/include/config 
                        /home/jarryd/project/openfpm_install_master/openfpm_pdata/include 
                        /home/jarryd/project/openfpm_install_master/openfpm_data/include 
                        /home/jarryd/project/openfpm_install_master/openfpm_vcluster/include 
                        /home/jarryd/project/openfpm_install_master/openfpm_io/include 
                        /home/jarryd/project/openfpm_install_master/openfpm_devices/include 
                        /home/jarryd/project/openfpm_dependencies_master/VCDEVEL/include 
                        /home/jarryd/project/openfpm_dependencies_master/METIS/include 
                        /home/jarryd/project/openfpm_dependencies_master/PARMETIS/include 
                        /home/jarryd/project/openfpm_dependencies_master/BOOST/include 
                        /home/jarryd/project/openfpm_dependencies_master/HDF5/include 
                        /home/jarryd/project/openfpm_dependencies_master/LIBHILBERT/include 
                        /home/jarryd/project/openfpm_dependencies_master/PETSC/include 
                        /home/jarryd/project/openfpm_dependencies_master/OPENBLAS/include 
                        /home/jarryd/project/openfpm_dependencies_master/SUITESPARSE/include 
                        /home/jarryd/project/openfpm_dependencies_master/EIGEN)


# Include MPI includes
include_directories(${MPI_INCLUDE_PATH})
if(MPI_COMPILE_FLAGS)
        set_target_properties(myProject PROPERTIES
                        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
        set_target_properties(tests PROPERTIES
                        LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()