#include <iostream>
#include <string>
#include <filesystem>

#include "util/PathsAndFiles.hpp"

#include "Grid/grid_dist_id.hpp"
#include "CSVReader/CSVReader.hpp"
#include "level_set/redistancing_Sussman/HelpFunctionsForGrid.hpp"

#include "../include/get_concentration_profile.hpp"

// For finding files to be analyzed
namespace fs = std::filesystem;

// Grid dimensions
const size_t dims = 2;

// Space indices
constexpr size_t x = 0, y = 1;

// Property indices
// CAREFUL: the number of properties and the grid size has to be the same as the grid that we want to load
constexpr size_t
		PHI_SDF                = 0,
		CONC_N                 = 1,
		CONC_NPLUS1            = 2,
		CONC_LAP               = 3,
		DIFFUSION_COEFFICIENT  = 4,
		K_SOURCE               = 5,
		K_SINK                 = 6;

typedef aggregate<double, double, double, double, double, double, double> props;

const std::string path_to_diffusion_result =
		"/home/jarryd/project/cms-cls-tea/diffusion/output_diffusion/";

// Parameters for the diffusion process
const double D = 0.1; // diffusion constant


/**
 * CAREFUL: When running same script several times without changing the path_output name, the values will just be
 * appended to the already existing csv file! So either change the name or delete csv file before running again
 */
 

int main(int argc, char* argv[])
{
	// Initialize library.
	openfpm_init(&argc, &argv);
	auto & v_cl = create_vcluster();
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Set current working directory, define output paths and create folders where output will be saved
	std::string cwd                     = get_cwd();
	const std::string path_output       = cwd + "/output_evaluation/";
	create_directory_if_not_exist(path_output);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Create grid of size NxN
	typedef double phi_type;
	size_t N = 64;
	const size_t sz[dims] = {N, N};
	const double radius = 1.0;
	const double box_lower = 0.0;
	const double box_upper = 4.0 * radius;
	Box<dims, double> box({box_lower, box_lower}, {box_upper, box_upper});
	Ghost<dims, long int> ghost(1);
	typedef grid_dist_id<dims, double, props > grid_type;
	grid_type g_dist(sz, box, ghost);
	g_dist.setPropNames({"PHI_SDF", "CONC_N", "CONC_NPLUS1", "CONC_LAP", "DIFFUSION_COEFFICIENT", "K_SOURCE",
	                     "K_SINK"});
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Load diffusion result
	// You can run the diffusion code in the diffusion folder and adapt the input path above accordingly
	/*
	g_dist.load(path_to_diffusion_result + "grid_diffuse_9900.hdf5");
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Get 1D concentration profile by summing substance over rays along the x-axis
	openfpm::vector<float> v_conc_slice;
	v_conc_slice.resize(N);
	openfpm::vector<float> v_distance_from_margin;
	v_distance_from_margin.resize(N);
	int x_start    = 0;
	int x_stop     = (int) N;
	const double y_margin = 0;
	
	// This is the function that we use
//	void get_1D_profile_from_2D(grid_type & grid,
//	                            const ny_type Ny,
//	                            const startstop_type x_start,
//	                            const startstop_type x_stop,
//	                            const Ty y_margin,
//	                            const scaling_type scaling,
//	                            openfpm::vector<distance_type> & v_distance_from_margin,
//	                            openfpm::vector<mass_type> & v_mass_slice
	

	// CAREFUL: When running same script several times without changing the path_output name, the values will just be
	// appended to the already existing csv file! So either change the name or delete csv file before running again
	
	get_1D_profile_from_2D<CONC_N>(g_dist,
	                               N,
	                               x_start,
	                               x_stop,
	                               y_margin,
								   1.0,
	                               v_distance_from_margin,
	                               v_conc_slice);
								   
	
	
    
	
	
		reduction_and_write_vectors_to_csv(v_distance_from_margin, v_conc_slice, path_output, "conc_versus_x.csv");
	*/
	/**
	 * This will produce a csv file with
	 * column 1: y-coordinate (= distance from y_margin)
	 * column 2: sum of concentration values over all grid points in a line along x-axis
	 *
	 * You can use the csv-file to plot the concentration profile using e.g. matlab or python matplotlib
	 */
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
	
	// Jarryd's edit to process all hdf5 files
	for (const auto & entry : fs::directory_iterator(path_to_diffusion_result)){
		const std::string file_name = entry.path();
        //std::cout << file_name.substr(file_name.size() - 4) << std::endl;
		
		if (file_name.substr(file_name.size() - 4) == "hdf5"){
			//std::cout << file_name.substr(file_name.size() - 4) << std::endl;
			
			g_dist.load(file_name);
	
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Get 1D concentration profile by summing substance over rays along the x-axis
			openfpm::vector<float> v_conc_slice;
			v_conc_slice.resize(N);
			openfpm::vector<float> v_distance_from_margin;
			v_distance_from_margin.resize(N);
			int x_start    = 0;
			int x_stop     = (int) N;
			const double y_margin = 0;
			
			get_1D_profile_from_2D<CONC_N>(g_dist,
										   N,
										   x_start,
										   x_stop,
										   y_margin,
										   1.0,
										   v_distance_from_margin,
										   v_conc_slice);
									   
			const std::string file_name_out = file_name.substr(path_to_diffusion_result.size(), file_name.size() - path_to_diffusion_result.size() - 5);
			std::cout << file_name_out << std::endl;
			reduction_and_write_vectors_to_csv(v_distance_from_margin, 
											   v_conc_slice, 
											   path_output, 
											   file_name_out + "_conc_versus_x.csv");
		}
	}
	
	
	openfpm_finalize();
	return 0;
}
