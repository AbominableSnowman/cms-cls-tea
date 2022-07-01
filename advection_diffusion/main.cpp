#include <iostream>
#include <algorithm>
#include <string>

#include "util/PathsAndFiles.hpp"
#include "level_set/redistancing_Sussman/AnalyticalSDF.hpp" // Analytical SDF to define the disk-shaped domain
#include "level_set/redistancing_Sussman/HelpFunctionsForGrid.hpp"
#include "level_set/redistancing_Sussman/RedistancingSussman.hpp" //for sussman redistancing
#include "FiniteDifference/Upwind_gradient.hpp"

#include "../include/FD_laplacian.hpp"
#include "../include/timesteps_stability.hpp"
#include "../include/Gaussian.hpp"
#include "../include/monitor_total_mass.hpp"


int main(int argc, char* argv[])
{
	openfpm_init(&argc, &argv); // Initialize library.
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Define output locations, advection + diffusion on/off, & what to save
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	bool save_vtk = true;
	bool save_hdf5 = false;
	bool save_mass = false;
	bool advection_on = false;
	bool diffusion_on = true;
	std::string cwd = get_cwd();
	const std::string path_output = cwd + "/output_advection_diffusion/";
	create_directory_if_not_exist(path_output);
	std::string save_path = path_output + "/";
	if (advection_on && diffusion_on) {save_path += "advection_diffusion";}
	else if (advection_on && !diffusion_on) {save_path += "advection";}
	else if (diffusion_on && !advection_on) {save_path += "diffusion";}
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Simulation & grid values
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Temporal values
	double t = 0;
	int iter = 0; // initial iteraton
	int max_iter = 1e2; // max iteration --> be careful, that the box is large enough to contain the growing disk!
	int interval_write = (int)(max_iter / 100); // set how many frames should be saved as vtk
	
	// Spatial values
	const size_t dims = 2; // Grid dimensions
	constexpr size_t x = 0, y = 1; // Space indices
	size_t N = 64; // Grid length
	const size_t sz[dims] = {N, N};
	const double radius = 1.0;
	const double box_lower = 0.0;
	const double box_upper = 4.0 * radius;
	const double center[dims] = {0.5*(box_upper+box_lower), 0.5*(box_upper+box_lower)};
	
	// Diffusion values
	const double D = 0.1; // diffusion constant
	double k_source = 1;
	double k_sink   = 1;
    double b_low = 0; // considered as embryo boundary
	double mu [dims]    = {box_upper/2.0, box_upper/2.0}; // For initial gaussian conc field
	double sigma [dims] = {box_upper/10.0, box_upper/10.0}; // For initial gaussian conc field
    double x_max = 0;
    double y_max = 0;
    double phi_max = 0;
	
	// Advection values
	const double v = 0.1; // velocity
	
	// Grid property indices		
	constexpr size_t
		CONC_N                 = 0,
		CONC_NPLUS1            = 1,
		CONC_LAP               = 2,
		DIFFUSION_COEFFICIENT  = 3,
		K_SOURCE               = 4,
		K_SINK                 = 5,
		PHI_N                  = 6,  // level-set function Phi
		PHI_NPLUS1             = 7,  // level-set function Phi of next timepoint
		V_SIGN                 = 8,  // sign of velocity, needed for the upwinding
		PHI_GRAD               = 9,  // gradient of phi (vector field)
		PHI_GRAD_MAGNITUDE     = 10; // Eucledian norm of gradient (scalar field)
			
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// OpenFPM Grid Initialization
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	auto & v_cl = create_vcluster();
	
	// Create grid of size NxN
	Box<dims, double> box({box_lower, box_lower}, {box_upper, box_upper});
	Ghost<dims, long int> ghost(1);
	typedef aggregate<double, double, double, 
					  double, double, double, 
					  double, double, int, 
					  double[dims], double> props;
	typedef grid_dist_id<dims, double, props > grid_type;
	grid_type g_dist(sz, box, ghost);
	g_dist.setPropNames({"CONC_N", "CONC_NPLUS1", "CONC_LAP", 
						 "DIFFUSION_COEFFICIENT", "K_SOURCE", "K_SINK", 
						 "PHI_N", "PHI_NPLUS1", "V_SIGN", 
						 "PHI_GRAD", "PHI_GRAD_MAGNITUDE"});
	
	// Grid Layer Inititializations
	init_grid_and_ghost<V_SIGN>(g_dist, 1);
	init_grid_and_ghost<PHI_N>(g_dist, -1);
	init_grid_and_ghost<PHI_NPLUS1>(g_dist, -1);
	init_grid_and_ghost<CONC_N>(g_dist, 0);
	init_grid_and_ghost<CONC_NPLUS1>(g_dist, 0);
	
	// Initialize level-set function with analytic signed distance function at each grid point
	init_analytic_sdf_circle<PHI_N>(g_dist, radius, center[x], center[y]);
	
	// Get the upwind gradient of Phi in order to get the surface normals (for advection)
	get_upwind_gradient<PHI_N, V_SIGN, PHI_GRAD>(g_dist, 1, true);
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Initialize Concentration Field
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// find the biggest SDF value and its location
	auto dom5 = g_dist.getDomainIterator();
	while(dom5.isNext()) // Loop over all grid points
	{
		auto key = dom5.get(); // index of current grid node
		if (g_dist.template get<PHI_N>(key) > phi_max){
            phi_max = g_dist.template get<PHI_N>(key);

            Point<grid_type::dims, typename grid_type::stype> coords = g_dist.getPos(key); 
            auto x = coords.get(0);
            auto y = coords.get(1);

            x_max = x;
            y_max = y;
        } 
            
		++dom5;
	}

    std::cout << "x is: " << x_max << std::endl;
    std::cout << "y is: " << y_max << std::endl;
    std::cout << "phi is: " << phi_max << std::endl;
	
	auto dom_conc = g_dist.getDomainIterator();
	typedef double phi_type;
	while(dom_conc.isNext()) // Loop over all grid points
	{
		auto key = dom_conc.get(); // index of current grid node
		Point<grid_type::dims, typename grid_type::stype> coords = g_dist.getPos(key); 
        // get coordinates of grid point
        auto x_coord = coords.get(0);
        auto y_coord = coords.get(1);
        if (g_dist.template getProp<PHI_N>(key) >= b_low - std::numeric_limits<phi_type>::epsilon()) {
            // Here, change if-condition accordingly to where you would like to have the source
            if(g_dist.template get<PHI_N>(key) < 0.15*phi_max && y_coord < y_max){
                g_dist.template get<K_SOURCE>(key) = k_source;
                g_dist.template get<K_SINK>(key) = 0;
            }
            else if (g_dist.template get<PHI_N>(key) < 0.15*phi_max && y_coord > y_max) {
                g_dist.template get<K_SOURCE>(key) = 0;
                g_dist.template get<K_SINK>(key) = k_sink;
            }
            else {
                g_dist.template get<K_SOURCE>(key) = 0;
                g_dist.template get<K_SINK>(key) = 0;
            }
        }
		++dom_conc;
	}
	// Check and print initial mass after initializing concentrations
	const double dx = g_dist.spacing(x), dy = g_dist.spacing(y); // Get grid spacings
	std::cout << "dx = " << dx << ", dy = " << dy << std::endl;
	double p_volume  = dx * dy;
	double m_initial = sum_prop_over_region<CONC_N, PHI_N>(g_dist, b_low) * p_volume; // Initial total mass
	std::cout << "m_initial : " << m_initial << std::endl;
	if (m_initial == 0){m_initial = 0.01;} // is this needed?
	// Save initial state
	g_dist.write(path_output + "grid_initial", FORMAT_BINARY);
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Stability Conditions
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Advection stability condition
	const double dt_adv = get_advection_time_step_cfl(g_dist, v, 0.1);
	std::cout << "dt (advection) = " << dt_adv << std::endl;
	// Diffusion stability condition
	const double dt_dif = get_diffusion_time_step(g_dist, D);
	std::cout << "dt (diffusion) = " << dt_dif << std::endl;
	// Final dt is min of the two:
	const double dt = std::min(dt_adv, dt_dif);
	
	
	
	
	
	
	
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Simulation
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	while(iter < max_iter) // looping over time
	{
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		if (diffusion_on){
			// Impose no-flux BCs at the disk interface
			auto dom4 = g_dist.getDomainIterator();
			while(dom4.isNext()) {
				auto key = dom4.get();
				if(g_dist.template getProp<PHI_N>(key) >= b_low - std::numeric_limits<phi_type>::epsilon()) {
					for(int d = 0; d < dims; ++d) {
						if(g_dist.template get<PHI_N>(key.move(d, 1)) < b_low + std::numeric_limits<double>::epsilon()) {
							g_dist.template get<CONC_N>(key.move(d, 1)) = g_dist.template get<CONC_N>(key);
						}
						
						if(g_dist.template get<PHI_N>(key.move(d, -1)) < b_low + std::numeric_limits<double>::epsilon()) {
							g_dist.template get<CONC_N>(key.move(d, -1)) = g_dist.template get<CONC_N>(key);
						}
					}
				}
				++dom4;
			}
			
			
			// Compute laplacian of concentration for whole grid
			get_laplacian_grid<CONC_N, CONC_LAP>(g_dist);

			// looping over gridpoints
			auto dom3 = g_dist.getDomainIterator();
			while (dom3.isNext()) {
				auto key = dom3.get();

				// diffusion only
				// g_dist.template get<CONC_NPLUS1>(key) = g_dist.template get<CONC_N>(key) + D * dt * g_dist.template get<CONC_LAP>(key);

				// reaction-diffusion
				// This is simple diffusion so far. For source and sink, you can add respective reaction terms
				if (g_dist.template getProp<PHI_N>(key) >= b_low - std::numeric_limits<phi_type>::epsilon())
				{
					g_dist.template get<CONC_NPLUS1>(key) = g_dist.template get<CONC_N>(key) + D * dt * g_dist.template get<CONC_LAP>(key) 
					+ dt*g_dist.template get<K_SOURCE>(key) - dt*g_dist.template get<K_SINK>(key)*g_dist.template get<CONC_N>(key);            
				}
				
				
				++dom3;
			}
			// Update CONC_N
			copy_gridTogrid<CONC_NPLUS1, CONC_N>(g_dist, g_dist);
		} // End diffusion
		
		if (advection_on){
			/** Advection: check condition of phi gradient. Count the number of points where PHI_GRAD > upper-bound 
			tolerance OR PHI_GRAD < lower-bound tolerance and if # of points out of range > tolerance value, then run 
			sussman redistancing to reinitialize phi.**/
			
			// ***Still need to incorporate the above***
			
			// Compute upwind gradient of phi for whole grid
			get_upwind_gradient<PHI_N, V_SIGN, PHI_GRAD>(g_dist, 1, true); // the upwind gradient is automatically
			// one-sided at the boundary
			get_vector_magnitude<PHI_GRAD, PHI_GRAD_MAGNITUDE, double>(g_dist);
			// Loop over grid and simulate growth using the surface normals (= magnitude gradient of phi) computed above
			// This just runs over the whole box so far
			
			auto dom = g_dist.getDomainIterator();
			auto key = dom.get();
			auto phi_gra_mag = g_dist.template get<PHI_GRAD_MAGNITUDE>(dom.get()); //get the magnitude of the phi gradient 
			std::cout << "phi_gra_mag:" << phi_gra_mag << std::endl;

			if (phi_gra_mag > 1.02)
			{	
				Redist_options<phi_type> redist_options;
				redist_options.min_iter                             = 1e3;
				redist_options.max_iter                             = 1e4;
		
				redist_options.convTolChange.value                  = 1e-7;
				redist_options.convTolChange.check                  = true;
				redist_options.convTolResidual.value                = 1e-6; // is ignored if convTolResidual.check = false;
				redist_options.convTolResidual.check                = false;
		
				redist_options.interval_check_convergence           = 1e3;
				redist_options.width_NB_in_grid_points              = 10;
				redist_options.print_current_iterChangeResidual     = true;
				redist_options.print_steadyState_iter               = true;
				redist_options.save_temp_grid                       = true;
				
				RedistancingSussman<grid_type, phi_type> redist_obj(g_dist, redist_options); // Instantiation of Sussman-redistancing class
				redist_obj.run_redistancing<PHI_N, PHI_N>(); //update phi_n using sussman redistancing
				get_upwind_gradient<PHI_N, V_SIGN, PHI_GRAD>(g_dist, 1, true); //calculate new gradient and magnitude using new phi_n
				get_vector_magnitude<PHI_GRAD, PHI_GRAD_MAGNITUDE, double>(g_dist);
				auto phi_gra_mag_2 = g_dist.template get<PHI_GRAD_MAGNITUDE>(dom.get()); //get the magnitude of the phi gradient 
				std::cout << "phi_gra_mag_2:" << phi_gra_mag_2 << std::endl;
			} 
			
			// Advection: Set PHI_NPLUS1 to PHI_N and update PHI_N
			while(dom.isNext()) // looping over the grid points
			{
				auto key = dom.get();
				g_dist.template get<PHI_NPLUS1>(key) = g_dist.template get<PHI_N>(key) + dt * v * g_dist.template get<PHI_GRAD_MAGNITUDE>(key);
				++dom;
			}
			// Update PHI_N
			copy_gridTogrid<PHI_NPLUS1, PHI_N>(g_dist, g_dist);
		} // End advection
		
		
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// End iteration; update iter, t, and save timestep
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Write grid to vtk and/or hdf5; save mass monitoring file
		if (iter % interval_write == 0){
			if (save_vtk){g_dist.write_frame(save_path, iter, FORMAT_BINARY);}
			if (save_hdf5){g_dist.save(save_path + std::to_string(iter) + ".hdf5");}
			//if (save_mass){monitor_absolute_mass_over_region<CONC_N, PHI_N>(g_dist, b_low, m_initial, p_volume, t, iter, path_output, save_path + "mass.csv");}
		}
		std::cout << "Time :" << t << std::endl;
		iter += 1;
		t += dt;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// End simulation
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	openfpm_finalize();
	return 0;
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