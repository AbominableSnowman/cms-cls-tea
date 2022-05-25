#include <iostream>

#include "util/PathsAndFiles.hpp"

#include "level_set/redistancing_Sussman/AnalyticalSDF.hpp" // Analytical SDF to define the disk-shaped diffusion domain
#include "level_set/redistancing_Sussman/HelpFunctionsForGrid.hpp"

#include "../include/FD_laplacian.hpp"
#include "../include/Gaussian.hpp"
#include "../include/timesteps_stability.hpp"


// Grid dimensions
const size_t dims = 2;

// Space indices
constexpr size_t x = 0, y = 1;

// Property indices
constexpr size_t
PHI_SDF                = 0,
CONC_N                 = 1,
CONC_NPLUS1            = 2,
CONC_LAP               = 3,
DIFFUSION_COEFFICIENT  = 4,
K_SOURCE               = 5,
K_SINK                 = 6;

typedef aggregate<double, double, double, double, double, double, double> props;



// Parameters for the diffusion process
const double D = 0.1; // diffusion constant


int main(int argc, char* argv[])
{
	// Initialize library.
	openfpm_init(&argc, &argv);
	auto & v_cl = create_vcluster();
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Set current working directory, define output paths and create folders where output will be saved
	std::string cwd                     = get_cwd();
	const std::string path_output       = cwd + "/output_diffusion/";
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
	const double center[dims] = {0.5*(box_upper+box_lower), 0.5*(box_upper+box_lower)};
	
	
	init_grid_and_ghost<CONC_N>(g_dist, 0); // Initialize grid and ghost layer with 0
	init_grid_and_ghost<CONC_NPLUS1>(g_dist, 0); // Initialize grid and ghost layer with 0
	init_grid_and_ghost<PHI_SDF>(g_dist, -1); // Initialize grid and ghost layer with -1
	
	// Initialize level-set function with analytic signed distance function at each grid point
	init_analytic_sdf_circle<PHI_SDF>(g_dist, radius, center[x], center[y]);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// For the gaussian shaped initial concentration
	double mu [dims]    = {box_upper/2.0, box_upper/2.0};
	double sigma [dims] = {box_upper/10.0, box_upper/10.0}; 
	
	// Initialize grid with initial concentration and diffusion coefficient
	auto dom = g_dist.getDomainIterator();
	while(dom.isNext()) // Loop over all grid points
	{
		auto key = dom.get(); // index of current grid node
		
		Point<grid_type::dims, typename grid_type::stype> coords = g_dist.getPos(key); // get coordinates of grid point
		
		g_dist.template get<CONC_N>(key)                = gaussian(coords, mu, sigma); // optional. alternatively set
		// to 0 and get initial concentration by adding the source term
		g_dist.template get<DIFFUSION_COEFFICIENT>(key) = D;
		
		++dom;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Initialize source and sink term
	double k_source = 1.0;
	double k_sink   = 0.1;
	
	auto dom2 = g_dist.getDomainIterator();
	while(dom2.isNext()) // Loop over all grid points
	{
		auto key = dom2.get(); // index of current grid node
		
		// Here, change if-condition accordingly to where you would like to have the source
		if(g_dist.template get<PHI_SDF>(key) > radius / 2.0)
		{
			g_dist.template get<K_SOURCE>(key) = k_source;
		}
		else
		{
			g_dist.template get<K_SOURCE>(key) = 0;
		}
		
		// Here, change if-condition accordingly to where you would like to have the sink
		if(g_dist.template get<PHI_SDF>(key) < radius / 2.0)
		{
			g_dist.template get<K_SINK>(key) = k_sink;
		}
		else
		{
			g_dist.template get<K_SINK>(key) = 0;
		}
		
		++dom2;
	}
	
	g_dist.write(path_output + "grid_initial", FORMAT_BINARY); // Save initial grid
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Get the diffusion timestep that fulfills the stability condition
	const double dx = g_dist.spacing(x), dy = g_dist.spacing(y); // if you want to know the grid spacing
	const double dt = get_diffusion_time_step(g_dist, D);
	std::cout << "dx = " << dx << ", dy = " << dy << ", dt = " << dt << std::endl;
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Diffusion using a forward-time central-space scheme
	double t = 0;
	int iter = 0; // initial iteraton
	int max_iter = 1e4; // max iteration
	int interval_write = (int)(max_iter / 100); // set how many frames should be saved as vtk
	while(iter < max_iter)
	{
		// Compute laplacian of concentration for whole grid
		get_laplacian_grid<CONC_N, CONC_LAP>(g_dist);
		
		// Loop over grid and run (reaction-)diffusion using the concentration laplacian computed above
		// THERE ARE NO BOUNDARY CONDITIONS FOR THE DIFFUSION DOMAIN YET
		// This just runs over the whole box so far
		auto dom3 = g_dist.getDomainIterator();
		while(dom3.isNext())
		{
			auto key = dom3.get();
			
			// This is simple diffusion so far. For source and sink, you can add respective reaction terms
			g_dist.template get<CONC_NPLUS1>(key) =
			        g_dist.template get<CONC_N>(key) + D * dt * g_dist.template get<CONC_LAP>(key);
			
			++dom3;
		}
		
		
		// Write grid to vtk
		if (iter % interval_write == 0)
		{
			g_dist.write_frame(path_output + "/grid_diffuse", iter, FORMAT_BINARY);
			std::cout << "Diffusion time :" << t << std::endl;
		}
		
		// Update CONC_N
		copy_gridTogrid<CONC_NPLUS1, CONC_N>(g_dist, g_dist);
		
		
		
		iter += 1;
		t += dt;
	}
	
	
	
	
	g_dist.save(path_output + "/grid_diffuse_" + std::to_string(iter) + ".hdf5"); // Save grid as hdf5 file which can
	// be reloaded for evaluation
	
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	openfpm_finalize();
	return 0;
}
