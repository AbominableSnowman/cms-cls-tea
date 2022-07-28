// source /Users/home/juwailes/openfpm_vars
// cmake ../. -DCMAKE_BUILD_TYPE=Release;

#include <iostream>
#include "util/PathsAndFiles.hpp"

#include "level_set/redistancing_Sussman/AnalyticalSDF.hpp" 
#include "level_set/redistancing_Sussman/HelpFunctionsForGrid.hpp"
#include "level_set/redistancing_Sussman/RedistancingSussman.hpp"

#include "FD_laplacian.hpp"
#include "Gaussian.hpp"
#include "timesteps_stability.hpp"
#include "monitor_total_mass.hpp"
#include "FiniteDifference/Upwind_gradient.hpp"

// set-up
const size_t dims = 2;
constexpr size_t x = 0, y = 1;

// speed and diffusion parameters
const double velocity_x = 1.75;
const double velocity_y = 3.75;
const double v[dims] = {velocity_x, velocity_y};
const double D = 0.9;
const double k_source = 1;
const double k_sink   = 1;
double b_low = 0;

// Property indices
constexpr size_t PHI_N = 0, PHI_NPLUS1 = 1, V_SIGN = 2, PHI_GRAD = 3, PHI_GRAD_MAGNITUDE = 4,
CONC_N = 5, CONC_NPLUS1 = 6, CONC_LAP = 7, K_SOURCE = 8, K_SINK = 9, CONC_N_GRAD = 10, VELOCITY = 11;

typedef aggregate<double, double, int, double[dims], double,
double, double, double, double, double, double[dims], double[dims]> props;

int main(int argc, char* argv[])
{
	// Initialize library.
	openfpm_init(&argc, &argv);
	auto & v_cl = create_vcluster();
	
	// Set current working directory, define output paths and create folders where output will be saved
	std::string cwd = get_cwd();
	const std::string path_output = cwd + "/output_diffusion/";
	create_directory_if_not_exist(path_output);
	
	// Grid creation
	typedef double phi_type;
	size_t N = 64;
	const size_t sz[dims] = {N, N};
	const double radius = 0.75;
	double rightTopCorner = 4.;
	double leftBottomCorner = 0;
	auto gridCenter = 0.5*(leftBottomCorner + rightTopCorner);
	const double center[dims] = {gridCenter, gridCenter};
	
	Box<dims, double> box({leftBottomCorner,leftBottomCorner}, {rightTopCorner,rightTopCorner});
	Ghost<dims, long int> ghost(1);
	typedef grid_dist_id<dims, double, props > grid_type;
	grid_type g_dist(sz, box, ghost);

	// Assigning names for readability
	g_dist.setPropNames({"PHI_N", "PHI_NPLUS1", "V_SIGN", "PHI_GRAD", "PHI_GRAD_MAGNITUDE",
	"CONC_N", "CONC_NPLUS1", "CONC_LAP", "K_SOURCE", "K_SINK", "CONC_N_GRAD", "VELOCITY"});
	
	// initializing grid per property with a value
	init_grid_and_ghost<CONC_N>(g_dist, 0);
	init_grid_and_ghost<CONC_NPLUS1>(g_dist, 0);
	init_grid_and_ghost<V_SIGN>(g_dist, 1);
	init_grid_and_ghost<PHI_N>(g_dist, -1);
	init_grid_and_ghost<PHI_NPLUS1>(g_dist, -1);
	init_analytic_sdf_circle<PHI_N>(g_dist, radius, center[x], center[y]);

    // stability condition
	auto const dt_diffusion = get_diffusion_time_step(g_dist, D);
	const double dt_advection = get_advection_time_step_cfl(g_dist, v[1], 0.1);
	auto dt = std::min(dt_advection, dt_diffusion);

	std::cout << "diffusion timestep is: " << dt_diffusion << std::endl;
	std::cout << "advection timestep is: " << dt_advection << std::endl;
	std::cout << "timestep is: " << dt << std::endl;

	// IC's: Gaussian 
	double mu [dims]    = {rightTopCorner/2. ,rightTopCorner/2.};
	double sigma [dims] = {rightTopCorner/5., rightTopCorner/5.}; 
	auto domGaus = g_dist.getDomainIterator();
	while(domGaus.isNext())
	{
		auto key = domGaus.get();
		Point<grid_type::dims, typename grid_type::stype> coords = g_dist.getPos(key);
		if (g_dist.template getProp<PHI_N>(key) >= b_low - std::numeric_limits<phi_type>::epsilon())
		{g_dist.template get<CONC_N>(key)= gaussian(coords, mu, sigma);}
		++domGaus;
	}

	g_dist.write(path_output + "grid_initial", FORMAT_BINARY); // Save initial grid
	
	// setting up the system for solving
	//////////////////////////////////////////////////////////////
	get_upwind_gradient<PHI_N, VELOCITY, PHI_GRAD>(g_dist, 1, true);
	get_upwind_gradient<CONC_N, VELOCITY, CONC_N_GRAD>(g_dist, 1, true);

	double t = 0;
	int iter = 0; 
	int max_iter = 1e2;
	int interval_write = (int)(max_iter / 100); 

	while(iter < max_iter)
	{	
		// Velocity assignment
		auto domVel = g_dist.getDomainIterator();
		while(domVel.isNext())
		{	
			auto key = domVel.get();
			auto phi_grad = g_dist.template get<PHI_GRAD>(key);
			for(size_t d = 0; d < dims; d++){g_dist.template get<VELOCITY>(key)[d] = phi_grad[d] * v[d];}
			++domVel;
		}

		get_upwind_gradient<PHI_N, VELOCITY, PHI_GRAD>(g_dist, 1, true);
		get_vector_magnitude<PHI_GRAD, PHI_GRAD_MAGNITUDE, double>(g_dist);
		
		int counter = 0;
		auto dz = g_dist.getSpacing();
		auto dx = dz.get(0);
		size_t narrowBand = 6;
		auto domRedistancing = g_dist.getDomainIterator();
		while(domRedistancing.isNext())
		{
			auto key = domRedistancing.get();
			auto phi_gra_mag = g_dist.template get<PHI_GRAD_MAGNITUDE>(key);
			if(abs(g_dist.template get<PHI_N>(key)) <= (narrowBand/2*dx))
			{
				counter = (phi_gra_mag > 1.15 || phi_gra_mag < 0.85) ? counter + 1 : counter; 
				// std::cout << "phi_gra_mag:" << phi_gra_mag << std::endl;
			}
			++domRedistancing;
		}
		if (counter > 20)
		{	
			Redist_options<phi_type> redist_options;
    		redist_options.min_iter                             = 1e3;
    		redist_options.max_iter                             = 1e4;
    
    		redist_options.convTolChange.value                  = 1e-7;
    		redist_options.convTolChange.check                  = true;
    		redist_options.convTolResidual.value                = 1e-6;
    		redist_options.convTolResidual.check                = false;
    
    		redist_options.interval_check_convergence           = 1e3;
    		redist_options.width_NB_in_grid_points              = 10;
    		redist_options.print_current_iterChangeResidual     = true;
    		redist_options.print_steadyState_iter               = true;
    		redist_options.save_temp_grid                       = true;
			
			RedistancingSussman<grid_type, phi_type> redist_obj(g_dist, redist_options);
			redist_obj.run_redistancing<PHI_N, PHI_N>();
			get_upwind_gradient<PHI_N, VELOCITY, PHI_GRAD>(g_dist, 1, true);
			get_vector_magnitude<PHI_GRAD, PHI_GRAD_MAGNITUDE, double>(g_dist);
		}

		// Solving the DE for the SDF 
		auto domSDF = g_dist.getDomainIterator();
		while(domSDF.isNext())
		{	
			auto key = domSDF.get();
			double growth_term = 0.;
			auto phi_grad = g_dist.template get<PHI_GRAD>(key);

			for(size_t d = 0; d < dims; d++){growth_term += phi_grad[d] * g_dist.template get<VELOCITY>(key)[d];}
			g_dist.template get<PHI_NPLUS1>(key) = g_dist.template get<PHI_N>(key) + dt * growth_term;

			++domSDF;
		}

		// impose no-flux BC's:
		auto domNoFlux = g_dist.getDomainIterator();
		while(domNoFlux.isNext()) {
			auto key = domNoFlux.get();
			if(g_dist.template getProp<PHI_NPLUS1>(key) >= b_low - std::numeric_limits<phi_type>::epsilon()) {
				for(int d = 0; d < dims; ++d) {
					if(g_dist.template get<PHI_NPLUS1>(key.move(d, 1)) < b_low + std::numeric_limits<double>::epsilon()) {
						g_dist.template get<CONC_N>(key.move(d, 1)) = g_dist.template get<CONC_N>(key);
					}
					
					if(g_dist.template get<PHI_NPLUS1>(key.move(d, -1)) < b_low + std::numeric_limits<double>::epsilon()) {
						g_dist.template get<CONC_N>(key.move(d, -1)) = g_dist.template get<CONC_N>(key);
					}
				}
			}
			++domNoFlux;
		}
		
		// compute delta.C
		get_laplacian_grid<CONC_N, CONC_LAP>(g_dist);

        // Solve the concentration DE
        auto domConDE = g_dist.getDomainIterator();
        while (domConDE.isNext())
		{
            auto key = domConDE.get();

			if (g_dist.template getProp<PHI_NPLUS1>(key) >= b_low - std::numeric_limits<phi_type>::epsilon())
			{	
				auto conc_grad = g_dist.template get<CONC_N_GRAD>(key);
				double advectionTerm = 0.;
				for(size_t d = 0; d < dims; d++){advectionTerm += conc_grad[d] * g_dist.template get<VELOCITY>(key)[d];}
				
            	g_dist.template get<CONC_NPLUS1>(key) = g_dist.template get<CONC_N>(key) + D * dt * g_dist.template get<CONC_LAP>(key) + dt * advectionTerm;
			}
            ++domConDE;
        }
		
		auto domCleanUp = g_dist.getDomainIterator();
        while (domCleanUp.isNext()) {
			auto key = domCleanUp.get();
			if (g_dist.template getProp<PHI_N>(key) <= b_low - std::numeric_limits<phi_type>::epsilon())
			{g_dist.template get<CONC_N>(key) = 0;}
			++domCleanUp;
		}

		// Write grid to vtk
		if (iter % interval_write == 0)
		{
			g_dist.write_frame(path_output + "/growth_and_diffusion", iter, FORMAT_BINARY);
			std::cout << "Time :" << t << std::endl;
		}
		
		// Update PHI_N and CON_N
		copy_gridTogrid<PHI_NPLUS1, PHI_N>(g_dist, g_dist);
		copy_gridTogrid<CONC_NPLUS1, CONC_N>(g_dist, g_dist);
		
		iter += 1;
		t += dt;
	}
	openfpm_finalize();
	return 0;
}
