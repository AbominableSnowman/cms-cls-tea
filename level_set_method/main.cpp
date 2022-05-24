#include <iostream>

#include "util/PathsAndFiles.hpp"

#include "level_set/redistancing_Sussman/AnalyticalSDF.hpp" // Analytical SDF to define the disk-shaped domain
#include "level_set/redistancing_Sussman/HelpFunctionsForGrid.hpp"

#include "FiniteDifference/Upwind_gradient.hpp"


// Grid dimensions
const size_t dims = 2;

// Space indices
constexpr size_t x = 0, y = 1;

// Property indices
constexpr size_t
		PHI_N                = 0, // level-set function Phi
		PHI_NPLUS1           = 1, // level-set function Phi of next timepoint
		V_SIGN               = 2, // sign of velocity, needed for the upwinding
		PHI_GRAD             = 3, // gradient of phi (vector field)
		PHI_GRAD_MAGNITUDE   = 4; // Eucledian norm of gradient (scalar field)

typedef aggregate<double, double, int, double[dims], double> props;



// Parameters for the growth process
const double v = 0.1; // velocity


int main(int argc, char* argv[])
{
	// Initialize library.
	openfpm_init(&argc, &argv);
	auto & v_cl = create_vcluster();
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Set current working directory, define output paths and create folders where output will be saved
	std::string cwd                     = get_cwd();
	const std::string path_output       = cwd + "/output_level_set/";
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
	g_dist.setPropNames({"PHI_N", "PHI_NPLUS1", "V_SIGN", "PHI_GRAD", "PHI_GRAD_MAGNITUDE"});
	const double center[dims] = {0.5*(box_upper+box_lower), 0.5*(box_upper+box_lower)};
	
	init_grid_and_ghost<V_SIGN>(g_dist, 1); // Initialize grid and ghost layer with 1
	init_grid_and_ghost<PHI_N>(g_dist, -1); // Initialize grid and ghost layer with -1
	init_grid_and_ghost<PHI_NPLUS1>(g_dist, -1); // Initialize grid and ghost layer with -1
	
	// Initialize level-set function with analytic signed distance function at each grid point
	init_analytic_sdf_circle<PHI_N>(g_dist, radius, center[x], center[y]);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Get timestep that fulfills the CFL condition
	const double dx = g_dist.spacing(x), dy = g_dist.spacing(y); // if you want to know the grid spacing
	const double dt = get_time_step_CFL(g_dist, v, 0.1);
	std::cout << "dx = " << dx << ", dy = " << dy << ", dt = " << dt << std::endl;
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Get the upwind gradient of Phi in order to get the surface normals
	get_upwind_gradient<PHI_N, V_SIGN, PHI_GRAD>(g_dist, 1, true);
	
	g_dist.write(path_output + "grid_initial", FORMAT_BINARY); // Save initial grid
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Now evolve level-set using forward-time and upwinding for the gradient
	double t = 0;
	int iter = 0; // initial iteraton
	int max_iter = 1e2; // max iteration --> be careful, that the box is large enough to contain the growing disk!
	int interval_write = (int)(max_iter / 100); // set how many frames should be saved as vtk
	while(iter < max_iter)
	{
		// Compute upwind gradient of phi for whole grid
		get_upwind_gradient<PHI_N, V_SIGN, PHI_GRAD>(g_dist, 1, true); // the upwind gradient is automatically
		// one-sided at the boundary
		get_vector_magnitude<PHI_GRAD, PHI_GRAD_MAGNITUDE, double>(g_dist);
		// Loop over grid and simulate growth using the surface normals (= magnitude gradient of phi) computed above
		// This just runs over the whole box so far
		auto dom = g_dist.getDomainIterator();
		while(dom.isNext())
		{
			auto key = dom.get();
			
			g_dist.template get<PHI_NPLUS1>(key) =
			        g_dist.template get<PHI_N>(key) + dt * v * g_dist.template get<PHI_GRAD_MAGNITUDE>(key);
			
			++dom;
		}
		
		
		// Write grid to vtk
		if (iter % interval_write == 0)
		{
			g_dist.write_frame(path_output + "/grid_growth", iter, FORMAT_BINARY);
			std::cout << "Time :" << t << std::endl;
		}
		
		// Update PHI_N
		copy_gridTogrid<PHI_NPLUS1, PHI_N>(g_dist, g_dist);
		
		
		
		iter += 1;
		t += dt;
	}
	
	/**
	 * 	 This is the simplest for of growth in normal direction
	 * 	 After some steps of growth (especially if not in normal direction), the signed distance feature of the
	 * 	 level-set function Phi will be distorted. Therefore, it has to be reinitialized from time to time. For this,
	 * 	 we want to use Sussman redistancing, which is implemented in OpenFPM. Examples and explanation, how to use
	 * 	 the Sussman redistancing, can be found here:
	 * 	 http://ppmcore.mpi-cbg.de/doxygen/openfpm/example_sussman_disk.html
	 *
	 *   How do you know, when to reinitialize? Check after how many iterations the phi gradient magnitude is very
	 *   different from 1.
	 */

	
	
	
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	openfpm_finalize();
	return 0;
}
