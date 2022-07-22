#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>

// For loading experiment parameters
#include "../include/nlohmann/json.hpp"
using json = nlohmann::json;

// OpenFPM standard library
#include "util/PathsAndFiles.hpp"
#include "level_set/redistancing_Sussman/AnalyticalSDF.hpp"
#include "level_set/redistancing_Sussman/HelpFunctionsForGrid.hpp"
#include "level_set/redistancing_Sussman/RedistancingSussman.hpp"
#include "FiniteDifference/Upwind_gradient.hpp"
// From Justina
#include "../include/FD_laplacian.hpp"
#include "../include/timesteps_stability.hpp"
#include "../include/Gaussian.hpp"
#include "../include/monitor_total_mass.hpp"

void yousefs_simulation(int experiment_n, json conditions);


int main(int argc, char* argv[])
{
	openfpm_init(&argc, &argv); // Initialize library.
	std::ifstream f("./testing_conditions.json");
	json conditions = json::parse(f);
	
	for(int experiment_n=1; experiment_n<=50; experiment_n++){
		yousefs_simulation(experiment_n, conditions);
		std::cout << "Experiment " << experiment_n << " finshed." << std::endl;
	}
	openfpm_finalize();
	return 0;
}





void yousefs_simulation(int experiment_n, json conditions)
{
	int diffusion_on = conditions[std::to_string(experiment_n)]["diffusion_on"];
	int growth_on = conditions[std::to_string(experiment_n)]["growth_on"];
	int advection_on = conditions[std::to_string(experiment_n)]["advection_on"];
	int source_sink_cond = conditions[std::to_string(experiment_n)]["source_sink_condition"];
	double D = conditions[std::to_string(experiment_n)]["diffusion_coefficient"]; 
	
	double velocity_magnitude = conditions[std::to_string(experiment_n)]["velocity_magnitude"];
	double v_x = std::sqrt(velocity_magnitude*velocity_magnitude/2);
	double v_y = std::sqrt(velocity_magnitude*velocity_magnitude/2);
	double v[2] = {v_x,v_y};
	//double t_max = conditions[std::to_string(experiment_n)]["t_max"];
	double t_max = 1.0;
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Define output locations & experiment values
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Outputs
	bool save_vtk = false;
	bool save_hdf5 = false;
	bool save_mass = false;
	bool save_csv = true;
	std::string output_folder = "/output_all_simulations/";
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Simulation & grid values
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Temporal values	
	
	
	
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
	double k_source = 1;
	double k_sink   = 1;
	double mu[dims]    = {box_upper/2.0, box_upper/2.0}; // For initial gaussian conc field
	double sigma[dims] = {box_upper/10.0, box_upper/10.0}; // For initial gaussian conc field
	
	// Signed distance function values
	double emb_boundary = 0; // embryo boundary: where SDF ~= 0
	typedef double phi_type;
	auto phi_epsilon = std::numeric_limits<phi_type>::epsilon();
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Initialize OpenFPM grid
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	auto & v_cl = create_vcluster();
	// Create grid of size NxN
	Box<dims, double> box({box_lower, box_lower}, {box_upper, box_upper});
	Ghost<dims, long int> ghost(1);
	
	// Grid property indices		
	constexpr size_t
							CONC_N                 = 0,
							CONC_NPLUS1            = 1,
							CONC_LAP               = 2,
							CONC_N_GRAD            = 3,
							K_SOURCE               = 4,
							K_SINK                 = 5,
							PHI_N                  = 6,  	// level-set function Phi
							PHI_NPLUS1             = 7,  	// level-set function Phi of next timepoint
							VELOCITY               = 8,  	// velocity vector field
							PHI_GRAD               = 9,  	
							PHI_GRAD_MAGNITUDE     = 10; 	// Eucledian norm of gradient (scalar field)
	typedef aggregate<  	double, 						// CONC_N
							double, 						// CONC_NPLUS1
							double, 						// CONC_LAP
							double[dims], 					// CONC_N_GRAD
							double, 						// K_SOURCE
							double, 						// K_SINK
							double, 						// PHI_N
							double, 						// PHI_NPLUS1
							double[dims], 					// VELOCITY
							double[dims], 					// PHI_GRAD
							double							// PHI_GRAD_MAGNITUDE
							> props;
	typedef grid_dist_id<dims, double, props > grid_type;
	grid_type g_dist(sz, box, ghost);
	g_dist.setPropNames({	"CONC_N", 
							"CONC_NPLUS1", 
							"CONC_LAP", 
							"CONC_N_GRAD", 
							"K_SOURCE", 
							"K_SINK", 
							"PHI_N", 
							"PHI_NPLUS1", 
							"VELOCITY", 
							"PHI_GRAD", 
							"PHI_GRAD_MAGNITUDE"});
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Initialize grid values
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	init_grid_and_ghost<PHI_N>(g_dist, -1);
	init_grid_and_ghost<PHI_NPLUS1>(g_dist, -1);
	init_grid_and_ghost<CONC_N>(g_dist, 0);
	init_grid_and_ghost<CONC_NPLUS1>(g_dist, 0);
	init_analytic_sdf_circle<PHI_N>(g_dist, radius, center[x], center[y]); // Level-set function
	
	if (growth_on || advection_on) // Phi gradient & gradient magnitude
	{
		get_upwind_gradient<PHI_N, VELOCITY, PHI_GRAD>(g_dist, 1, true);
		get_vector_magnitude<PHI_GRAD, PHI_GRAD_MAGNITUDE, double>(g_dist);
	}
	
	// For source/sink: find the biggest phi value and its location + max x/y coordinates in domain
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
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Concentration initial condition, source/sink, & velocity vector field
	auto init_dom_iter = g_dist.getDomainIterator();
	while(init_dom_iter.isNext()) {
		auto key = init_dom_iter.get(); // index of current grid node
		auto phi_here = g_dist.template getProp<PHI_N>(key);
		Point<grid_type::dims, typename grid_type::stype> coords = g_dist.getPos(key);
		
		// Velocity. Defined everywhere b/c of growing domain
		if (growth_on || advection_on){
			auto phi_grad_mag = g_dist.template get<PHI_GRAD_MAGNITUDE>(key);
			auto phi_grad = g_dist.template getProp<PHI_GRAD>(key);
			// Calculate unit vector of phi gradient to get correct magnitude & direction of velocity
			// Vector multiplication in case we want unequal velocities between x & y
			double v_adjusted[dims] = {v[0]*phi_grad[0], v[1]*phi_grad[1]/phi_grad_mag};
			g_dist.template getProp<VELOCITY>(key)[0] = v_adjusted[0];
			g_dist.template getProp<VELOCITY>(key)[1] = v_adjusted[1];
		}
		
		if (phi_here >= emb_boundary - phi_epsilon) {
			////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Concentration initial conditions
			if (source_sink_cond == 0){
				g_dist.template get<CONC_N>(key) = gaussian(coords, mu, sigma);
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
	double p_volume  = dx * dy;
	double m_initial = std::max(sum_prop_over_region<CONC_N, PHI_N>(g_dist, emb_boundary) * p_volume, 0.01); // Initial total mass
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Stability Conditions
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Diffusion stability condition
	const double dt_dif = get_diffusion_time_step(g_dist, D);
	const double dt = dt_dif;
	if (growth_on || advection_on){
		grid_type::stype v_grid_type[dims] = {v[0], v[1]}; // need to make grid type v for function to work
		double dt_adv = get_advection_time_step_cfl(g_dist, v_grid_type, 0.1);
		const double dt = std::min(dt_adv, dt_dif); // Final dt is min of the two
	}
	int max_iter = (int)(t_max / dt) + 1;
	int interval_write = std::round(max_iter / 100); // set how many frames should be saved
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Configure outputs and save initial conditions
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	std::string cwd = get_cwd();
	const std::string path_output = cwd + output_folder;
	create_directory_if_not_exist(path_output);
	std::string save_name = "sim_" + std::to_string(experiment_n);
	
	// Initialize csv writer
	std::ofstream file_out;
	std::string csv_row = "";
	if (save_csv){ // but don't create file unless save_csv is on
		std::string csv_path_output = path_output + save_name + ".csv";
		create_file_if_not_exist(csv_path_output);
		file_out.open(csv_path_output, std::ios_base::app);
		file_out << // Fill first two rows with simulation settings
		"diffusion," 							<< 
		"growth," 								<< 
		"advection," 							<<
		"source_sink_condition," 				<<
		"max_iter," 							<< 
		"interval_write," 						<< 
		"grid_length," 							<<
		"diffusion_coefficient," 				<< 
		"source_val," 							<< 
		"sink_val," 							<<
		"v_x," 									<< 
		"v_y," 									<< 
		"dt," 									<< 
		"t_max,"								<<
		"dx," 									<<
		"dy" 									<< "\n" <<
		
		diffusion_on 							<< "," << 
		growth_on 								<< "," << 
		advection_on 							<< "," << 
		source_sink_cond 						<< "," <<
		max_iter 								<< "," << 
		interval_write 							<< "," << 
		N 										<< "," <<
		to_string_with_precision(D, 3) 			<< "," << 
		to_string_with_precision(k_source, 3) 	<< "," << 
		to_string_with_precision(k_sink, 3) 	<< "," <<
		to_string_with_precision(v[0], 3) 		<< "," << 
		to_string_with_precision(v[1], 3) 		<< "," << 
		to_string_with_precision(dt, 6) 		<< "," << 
		to_string_with_precision(t_max, 6) 		<< "," << 
		to_string_with_precision(dx, 6) 		<< "," << 
		to_string_with_precision(dy, 6) 		<< "\n\n";
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
		// Update field calculations for whole grid; don't need to update every loop for advection
		if (growth_on){
			get_upwind_gradient<PHI_N, VELOCITY, PHI_GRAD>(g_dist, 1, true);
			get_vector_magnitude<PHI_GRAD, PHI_GRAD_MAGNITUDE, double>(g_dist);
		}
		if (diffusion_on){
			get_laplacian_grid<CONC_N, CONC_LAP>(g_dist);
			get_upwind_gradient<CONC_N, VELOCITY, CONC_N_GRAD>(g_dist, 1, true);
		}
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Grid maintenance: no-flux boundary, source/sink expansion, SDF distortion
		auto maintenance_dom_iter = g_dist.getDomainIterator();
		int phi_grad_tol_break = 0; // counts the # of grid points outside of tolerance range
		while(maintenance_dom_iter.isNext()) {
			auto key = maintenance_dom_iter.get();
			auto phi_here = g_dist.template getProp<PHI_N>(key);
			auto conc_here = g_dist.template get<CONC_N>(key);
			
			// Save row to csv
			if (save_csv && iter % interval_write == 0){
				csv_row += "," + to_string_with_precision(conc_here);
			}
				
			// No-flux boundary + source/sink expansion
			if(phi_here >= emb_boundary - phi_epsilon) {
				for(int d = 0; d < dims; ++d) {
					if(g_dist.template get<PHI_N>(key.move(d, 1)) < emb_boundary + phi_epsilon) {
						// Impose no-flux boundary
						g_dist.template get<CONC_N>(key.move(d, 1)) = conc_here;
						
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
						g_dist.template get<CONC_N>(key.move(d, -1)) = conc_here;
						
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
					if (phi_grad_mag > 1.15 || phi_grad_mag < 0.8){
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
						auto grad_u = g_dist.template get<CONC_N_GRAD>(key);
						auto v_here = g_dist.template get<VELOCITY>(key);
						double v_advec = 0;
						// Dot product with concentration gradient to get final advection term						
						for(size_t d = 0; d < dims; d++){v_advec += v_here[d] * grad_u[d];}
						g_dist.template get<CONC_NPLUS1>(key) = u+ D*dt*laplacian_u + dt*ksource_here 
																- dt*ksink_here*u + dt*v_advec;
					}
				}
				++diff_dom_iter;
			}
			copy_gridTogrid<CONC_NPLUS1, CONC_N>(g_dist, g_dist); // Update CONC_N
		} // End diffusion
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		if (growth_on){
			if (phi_grad_tol_break > 10) {	
				Redist_options<phi_type> redist_options;
				redist_options.min_iter                             = 1e3;
				redist_options.max_iter                             = 1e4;
				redist_options.convTolChange.value                  = 1e-7;
				redist_options.convTolChange.check                  = true;
				redist_options.convTolResidual.value                = 1e-6;//is ignored if convTolResidual.check=false;
				redist_options.convTolResidual.check                = false;
				redist_options.interval_check_convergence           = 1e3;
				redist_options.width_NB_in_grid_points              = 10;
				redist_options.print_current_iterChangeResidual     = false;
				redist_options.print_steadyState_iter               = true;
				redist_options.save_temp_grid                       = false;
				
				// Redistancing
				RedistancingSussman<grid_type, phi_type> redist_obj(g_dist, redist_options); 
				redist_obj.run_redistancing<PHI_N, PHI_N>();
				// Calculate new gradient and magnitude using new phi_n
				get_upwind_gradient<PHI_N, VELOCITY, PHI_GRAD>(g_dist, 1, true); 
				get_vector_magnitude<PHI_GRAD, PHI_GRAD_MAGNITUDE, double>(g_dist);
			}
			
			// Set PHI_NPLUS1 to PHI_N and update PHI_N
			auto phi_dom_iter = g_dist.getDomainIterator();
			while(phi_dom_iter.isNext()) // looping over the grid points
			{
				auto key = phi_dom_iter.get();
				
				auto phi_mag = g_dist.template getProp<PHI_GRAD_MAGNITUDE>(key);
				g_dist.template get<PHI_NPLUS1>(key) = g_dist.template get<PHI_N>(key) + dt * v[0] * phi_mag;
				
				
				/*
				double dot_v_dphi = 0;
				auto v_here = g_dist.template get<VELOCITY>(key);
				auto grad_phi = g_dist.template getProp<PHI_GRAD>(key);
				for(size_t d = 0; d < dims; d++){dot_v_dphi += v_here[d] * grad_phi[d];}
				g_dist.template get<PHI_NPLUS1>(key) = g_dist.template get<PHI_N>(key) + dt * dot_v_dphi;
				*/
				
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
			if (save_vtk){g_dist.write_frame(path_output + save_name, iter, FORMAT_BINARY);}
			if (save_hdf5){g_dist.save(path_output + save_name + std::to_string(iter) + ".hdf5");}
			if (save_mass){monitor_absolute_mass_over_region<CONC_N, PHI_N>(g_dist, emb_boundary, m_initial, p_volume, t, iter, path_output, save_name + "_mass.csv");}
			if (save_csv){file_out << csv_row << std::endl;}
		}
		iter += 1;
		t += dt;
	}
	file_out.close();
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// End simulation
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}