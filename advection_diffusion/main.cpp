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
	// Define output locations & experiment values
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	bool diffusion_on = true;
	bool growth_on = false;
	bool advection_on = true;
	bool gaussian_ic = true;
	int source_sink_cond = 0; // 1-4 different testing scenarios for source & sink locations
	/*	0: No sources or sinks
		1: Source on bottom, sink on top
		2: Sources and sinks covering the entire disk
		3: Source in center, sink on boundary
		4: Sources and sinks covering either side of embryo (separated across x-axis)*/
	
	// Output locations
	bool save_vtk = true;
	bool save_hdf5 = false;
	bool save_mass = false;
	bool save_csv = false;
	std::string cwd = get_cwd();
	const std::string path_output = cwd + "/output_advection_diffusion/";
	create_directory_if_not_exist(path_output);
	std::string save_path = path_output + "SrcSnk" + std::to_string(source_sink_cond);
	if (diffusion_on){save_path += "_diff";}
	if (growth_on){save_path += "_growth";}
	if (advection_on){save_path += "_advec";}
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Simulation & grid values
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Temporal values
	int max_iter = 1e3; // max iteration --> be careful, that the box is large enough to contain the growing disk!
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
	double mu[dims]    = {box_upper/2.0, box_upper/2.0}; // For initial gaussian conc field
	double sigma[dims] = {box_upper/10.0, box_upper/10.0}; // For initial gaussian conc field
	
	// Growth values
	const double v[dims] = {0.2, 0.2};
	
	// Grid property indices		
	constexpr size_t
		CONC_N                 = 0,
		CONC_NPLUS1            = 1,
		CONC_LAP               = 2,
		CONC_N_GRAD            = 3,
		K_SOURCE               = 4,
		K_SINK                 = 5,
		PHI_N                  = 6,  // level-set function Phi
		PHI_NPLUS1             = 7,  // level-set function Phi of next timepoint
		V_SIGN                 = 8,  // sign of velocity, needed for the upwinding
		PHI_GRAD               = 9,  // gradient of phi (vector field)
		PHI_GRAD_MAGNITUDE     = 10; // Eucledian norm of gradient (scalar field)
	
	// Signed distance function values
	double emb_boundary = 0; // embryo boundary: where SDF ~= 0
	typedef double phi_type;
	auto phi_epsilon = std::numeric_limits<phi_type>::epsilon();
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// OpenFPM Grid Initialization
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	auto & v_cl = create_vcluster();
	
	// Create grid of size NxN
	Box<dims, double> box({box_lower, box_lower}, {box_upper, box_upper});
	Ghost<dims, long int> ghost(1);
	typedef aggregate<double, double, double, 
					  double[dims], double, double, 
					  double, double, int, 
					  double[dims], double> props;
	typedef grid_dist_id<dims, double, props > grid_type;
	grid_type g_dist(sz, box, ghost);
	g_dist.setPropNames({"CONC_N", "CONC_NPLUS1", "CONC_LAP", 
						 "CONC_N_GRAD", "K_SOURCE", "K_SINK", 
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
	// Initialize source, sink, & concentration field
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Find the biggest phi value and its location + max x/y coordinates in embryo
	double phi_max = 0;
	double x_max = 0;
    double y_max = 0;
	auto dom_phi_max = g_dist.getDomainIterator();
	while(dom_phi_max.isNext()) // Loop over all grid points
	{
		auto key = dom_phi_max.get(); // index of current grid node
		if (g_dist.template get<PHI_N>(key) > phi_max){
			auto phi_max_key = key;
            phi_max = g_dist.template get<PHI_N>(key);

            Point<grid_type::dims, typename grid_type::stype> coords = g_dist.getPos(key); 
            auto x = coords.get(0);
            auto y = coords.get(1);

            x_max = x;
            y_max = y;
        } 
            
		++dom_phi_max;
	}
	
    std::cout << "x is: " << x_max << std::endl;
    std::cout << "y is: " << y_max << std::endl;
    std::cout << "phi is: " << phi_max << std::endl;
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Set source & sink locations & initialize concentration field
	auto init_dom_iter = g_dist.getDomainIterator();
	while(init_dom_iter.isNext()) {
		auto key = init_dom_iter.get(); // index of current grid node
		auto phi_here = g_dist.template getProp<PHI_N>(key);
		Point<grid_type::dims, typename grid_type::stype> coords = g_dist.getPos(key);
		
		if (phi_here >= emb_boundary - phi_epsilon) {
			////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Concentration initial conditions
			if (gaussian_ic){
				g_dist.template get<CONC_N>(key)= gaussian(coords, mu, sigma);
			}
			////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Sources and sinks
			if (source_sink_cond == 1){
				auto x_coord = coords.get(0);
				auto y_coord = coords.get(1);
				
				if(phi_here < 0.15*phi_max && y_coord < y_max){
					g_dist.template get<K_SOURCE>(key) = k_source;
					g_dist.template get<K_SINK>(key) = 0;
				}
				else if (phi_here < 0.15*phi_max && y_coord > y_max) {
					g_dist.template get<K_SOURCE>(key) = 0;
					g_dist.template get<K_SINK>(key) = k_sink;
				}
				else {
					g_dist.template get<K_SOURCE>(key) = 0;
					g_dist.template get<K_SINK>(key) = 0;
				}
			}
			else if (source_sink_cond == 2){
				g_dist.template get<K_SOURCE>(key) = k_source;
				g_dist.template get<K_SINK>(key) = k_sink;
			}
			else if (source_sink_cond == 3){
				if(phi_here < 0.1*phi_max){
					g_dist.template get<K_SOURCE>(key) = 0;
					g_dist.template get<K_SINK>(key) = k_sink;
				}
				else if (phi_here > 0.75*phi_max){
					g_dist.template get<K_SOURCE>(key) = k_source;
					g_dist.template get<K_SINK>(key) = 0;
				}
				else {
					g_dist.template get<K_SOURCE>(key) = 0;
					g_dist.template get<K_SINK>(key) = 0;
				}
			}
			// Scenario 4
			else if (source_sink_cond == 4){
				auto x_coord = coords.get(0);
				auto y_coord = coords.get(1);
				
				if (y_coord < center[y]){
					g_dist.template get<K_SOURCE>(key) = k_source;
					g_dist.template get<K_SINK>(key) = 0;
				}
				else if (y_coord > center[y]){
					g_dist.template get<K_SOURCE>(key) = 0;
					g_dist.template get<K_SINK>(key) = k_sink;
				}
				else {
					g_dist.template get<K_SOURCE>(key) = 0;
					g_dist.template get<K_SINK>(key) = 0;
				}
			}
			////////////////////////////////////////////////////////////////////////////////////////////////////////////
        }
		++init_dom_iter;
	}
	
	// Check and print initial mass after initializing concentrations
	const double dx = g_dist.spacing(x), dy = g_dist.spacing(y); // Get grid spacings
	std::cout << "dx = " << dx << ", dy = " << dy << std::endl;
	double p_volume  = dx * dy;
	double m_initial = sum_prop_over_region<CONC_N, PHI_N>(g_dist, emb_boundary) * p_volume; // Initial total mass
	std::cout << "m_initial : " << m_initial << std::endl;
	if (m_initial == 0){m_initial = 0.01;} // is this needed?
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Stability Conditions
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Diffusion stability condition
	const double dt_dif = get_diffusion_time_step(g_dist, D);
	const double dt = dt_dif;
	if (growth_on || advection_on){
		//auto max_v = *std::max_element(std::begin(v), std::end(v));
		//const double dt_adv = get_advection_time_step_cfl(g_dist, max_v, 0.1);
		double sum = 0;
		for (size_t d = 0; d < dims; d++)
		{
			sum += v[d] / g_dist.spacing(d);
		}
		const double dt_adv = 0.1 / abs(sum);
		
		std::cout << "dt (advection) = " << dt_adv << std::endl;
		std::cout << "dt (diffusion) = " << dt_dif << std::endl;
		// Final dt is min of the two
		const double dt = std::min(dt_adv, dt_dif);
	}
	std::cout << "dt = " << dt << std::endl;
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Saving
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Save initial state
	if (save_vtk){g_dist.write(path_output + "grid_initial", FORMAT_BINARY);}
	// Initialize csv writer
	std::ofstream file_out;
	std::string csv_row = "";
	if (save_csv){
		// but don't create file unless save_csv is on
		std::string csv_path_output = save_path + ".csv";
		create_file_if_not_exist(csv_path_output);
		file_out.open(csv_path_output, std::ios_base::app); // append instead of overwrite
	}
	
	
	
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Simulation
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	double t = 0;
	int iter = 0;
	while(iter < max_iter) // looping over time
	{
		// start of new row in csv file
		if (save_csv && iter % interval_write == 0){csv_row = to_string_with_precision(t, 6);} 
		// Update field calculations for whole grid
		if (growth_on || advection_on){
			get_upwind_gradient<PHI_N, V_SIGN, PHI_GRAD>(g_dist, 1, true);
			get_vector_magnitude<PHI_GRAD, PHI_GRAD_MAGNITUDE, double>(g_dist);
		}
		if (diffusion_on){
			get_laplacian_grid<CONC_N, CONC_LAP>(g_dist);
			get_upwind_gradient<CONC_N, V_SIGN, CONC_N_GRAD>(g_dist, 1, true);
		}
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Grid maintenance: no-flux boundary, source/sink expansion, SDF distortion
		auto maintenance_dom_iter = g_dist.getDomainIterator();
		int phi_grad_tol_break = 0; // counts the # of grid points outside of tolerance range
		while(maintenance_dom_iter.isNext()) {
			auto key = maintenance_dom_iter.get();
			auto phi_here = g_dist.template getProp<PHI_N>(key);
			if (save_csv && iter % interval_write == 0){
				csv_row += "," + to_string_with_precision(g_dist.template get<CONC_N>(key));
				}
			if(phi_here >= emb_boundary - phi_epsilon) {
				for(int d = 0; d < dims; ++d) {
					if(g_dist.template get<PHI_N>(key.move(d, 1)) < emb_boundary + phi_epsilon) {
						// Impose no-flux boundary
						g_dist.template get<CONC_N>(key.move(d, 1)) = g_dist.template get<CONC_N>(key);
						
						if (growth_on){
							// Expand sources / sinks
							if (g_dist.template get<K_SOURCE>(key) == k_source){
								g_dist.template get<K_SOURCE>(key.move(d, 1)) = k_source;
							}
							if (g_dist.template get<K_SINK>(key) == k_sink){
								g_dist.template get<K_SINK>(key.move(d, 1)) = k_sink;
							}
						}
					}
					if(g_dist.template get<PHI_N>(key.move(d, -1)) < emb_boundary + phi_epsilon) {
						// Impose no-flux boundary
						g_dist.template get<CONC_N>(key.move(d, -1)) = g_dist.template get<CONC_N>(key);
						
						if (growth_on){
							// Expand sources / sinks
							if (g_dist.template get<K_SOURCE>(key) == k_source){
								g_dist.template get<K_SOURCE>(key.move(d, -1)) = k_source;
							}
							if (g_dist.template get<K_SINK>(key) == k_sink){
								g_dist.template get<K_SINK>(key.move(d, -1)) = k_sink;
							}	
						}
					}
				}
				////////////////////////////////////////////////////////////////////////////////////////////////////////
				// Determine if redistancing needs to be run based on # of grid points 
				if (growth_on){
					auto phi_grad_mag = g_dist.template get<PHI_GRAD_MAGNITUDE>(key); //get the magnitude of the phi gradient
					if (phi_grad_mag > 1.2 || phi_grad_mag < 0.8){
						phi_grad_tol_break += 1;
					}
				}
			}
			++maintenance_dom_iter;
		}
		
		
		
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		if (diffusion_on){
			// Update grid points by solving ODE 
			auto diff_dom_iter = g_dist.getDomainIterator();
			while (diff_dom_iter.isNext()) {
				auto key = diff_dom_iter.get();
				auto phi_here = g_dist.template getProp<PHI_N>(key);
				auto u = g_dist.template get<CONC_N>(key);
				auto laplacian_u = g_dist.template get<CONC_LAP>(key);
				auto epsilon_phi = std::numeric_limits<phi_type>::epsilon();
				auto ksource_here = g_dist.template get<K_SOURCE>(key);
				auto ksink_here = g_dist.template get<K_SINK>(key);
				
				if (phi_here >= emb_boundary - epsilon_phi){
					
					if (diffusion_on && !advection_on){
						g_dist.template get<CONC_NPLUS1>(key) = u + D*dt*laplacian_u 
																+ dt*ksource_here - dt*ksink_here*u;            
					}
					else if (diffusion_on && advection_on){
						// Get unit vector of phi gradient to correct direction of velocity
						auto phi_grad_mag = g_dist.template get<PHI_GRAD_MAGNITUDE>(key);
						auto phi_grad = g_dist.template getProp<PHI_GRAD>(key);
						// vector multiplication in case we want unequal velocities between x&y
						double v_adjusted[dims] = {v[0]*phi_grad[0]/phi_grad_mag, v[1]*phi_grad[1]/phi_grad_mag};
						// Dot product with concentration gradient to get final advection term
						auto grad_u = g_dist.template get<CONC_N_GRAD>(key);
						double v_advec = 0;
						for(size_t d = 0; d < dims; d++){v_advec += v_adjusted[d] * grad_u[d];}
						g_dist.template get<CONC_NPLUS1>(key) = u+ D*dt*laplacian_u + dt*ksource_here 
																- dt*ksink_here*u + dt*v_advec;
					}
				}
				++diff_dom_iter;
			}
			copy_gridTogrid<CONC_NPLUS1, CONC_N>(g_dist, g_dist); // Update CONC_N
		} // End diffusion
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		if (growth_on){
			if (phi_grad_tol_break > 20) {	
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
				
				// Instantiation of Sussman-redistancing class
				RedistancingSussman<grid_type, phi_type> redist_obj(g_dist, redist_options); 
				//Update SDF using Sussman Redistancing 
				redist_obj.run_redistancing<PHI_N, PHI_N>(); 
				// Calculate new gradient and magnitude using new phi_n
				get_upwind_gradient<PHI_N, V_SIGN, PHI_GRAD>(g_dist, 1, true); 
				get_vector_magnitude<PHI_GRAD, PHI_GRAD_MAGNITUDE, double>(g_dist);
			}
			
			// Set PHI_NPLUS1 to PHI_N and update PHI_N
			auto phi_dom_iter = g_dist.getDomainIterator();
			while(phi_dom_iter.isNext()) // looping over the grid points
			{
				auto key = phi_dom_iter.get();
				g_dist.template get<PHI_NPLUS1>(key) = g_dist.template get<PHI_N>(key) + dt * v[0] * g_dist.template get<PHI_GRAD_MAGNITUDE>(key);
				++phi_dom_iter;
			}
			// Update PHI_N
			copy_gridTogrid<PHI_NPLUS1, PHI_N>(g_dist, g_dist);
		} // End growth
		
		
		
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// End iteration; update iter, t, and save timestep
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Write grid to vtk and/or hdf5; save mass monitoring file
		if (iter % interval_write == 0){
			if (save_vtk){g_dist.write_frame(save_path, iter, FORMAT_BINARY);}
			if (save_hdf5){g_dist.save(save_path + std::to_string(iter) + ".hdf5");}
			//if (save_mass){monitor_absolute_mass_over_region<CONC_N, PHI_N>(g_dist, emb_boundary, m_initial, p_volume, t, iter, path_output, save_path + "mass.csv");}
			if (save_csv){file_out << csv_row << std::endl;}
		}
		std::cout << "Time :" << t << ", iteration: " << std::to_string(iter) << std::endl;
		iter += 1;
		t += dt;
	}
	file_out.close();
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