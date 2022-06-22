#include <iostream>

#include "util/PathsAndFiles.hpp"

// these are available in openfpm_data and openfpm_install_master
// Analytical SDF to define the disk-shaped diffusion domain
#include "level_set/redistancing_Sussman/AnalyticalSDF.hpp" 
#include "level_set/redistancing_Sussman/HelpFunctionsForGrid.hpp"

// where are these?
// no in dependencies or install
#include "FD_laplacian.hpp"
#include "Gaussian.hpp"
#include "timesteps_stability.hpp"
#include "monitor_total_mass.hpp"

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
	g_dist.setPropNames({"PHI_SDF", "CONC_N", "CONC_NPLUS1", "CONC_LAP", "DIFFUSION_COEFFICIENT", "K_SOURCE", "K_SINK"});
	const double center[dims] = {0.5*(box_upper+box_lower), 0.5*(box_upper+box_lower)};
	
	
	init_grid_and_ghost<CONC_N>(g_dist, 0); // Initialize grid and ghost layer with 0
	init_grid_and_ghost<CONC_NPLUS1>(g_dist, 0); // Initialize grid and ghost layer with 0
	init_grid_and_ghost<PHI_SDF>(g_dist, -1); // Initialize grid and ghost layer with -1
	
	// Initialize level-set function with analytic signed distance function at each grid point
	init_analytic_sdf_circle<PHI_SDF>(g_dist, radius, center[x], center[y]);	

    // stability condition
	auto dx = g_dist.spacing(x), dy = g_dist.spacing(y);
	auto dt = get_diffusion_time_step(g_dist, D);
	std::cout << "dx = " << dx << ", dy = " << dy << ", dt = " << dt << std::endl;
	

    // Initialize source and sink term
	double k_source = 1;
	double k_sink   = 1;

	// considered as embryo boundary
    double b_low = 0; 
	
	// Gaussian 
	double mu [dims]    = {box_upper/2.0, box_upper/2.0};
	double sigma [dims] = {box_upper/10.0, box_upper/10.0}; 


    double x_max = 0;
    double y_max = 0;
    double phi_max = 0;

    // find the biggest SDF value and its location
	auto dom5 = g_dist.getDomainIterator();
	while(dom5.isNext()) // Loop over all grid points
	{
		auto key = dom5.get(); // index of current grid node
		if (g_dist.template get<PHI_SDF>(key) > phi_max){
            phi_max = g_dist.template get<PHI_SDF>(key);

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

    std::cout << std::endl; 

	// Initialize grid with initial concentration and diffusion coefficient
	auto dom = g_dist.getDomainIterator();
	while(dom.isNext()) // Loop over all grid points
	{
		auto key = dom.get(); // index of current grid node
		
		//g_dist.template get<CONC_N>(key)= gaussian(coords, mu, sigma);
		//g_dist.template get<DIFFUSION_COEFFICIENT>(key) = D;

		Point<grid_type::dims, typename grid_type::stype> coords = g_dist.getPos(key); 
        // get coordinates of grid point
        auto x = coords.get(0);
        auto y = coords.get(1);
		
        if (g_dist.template getProp<PHI_SDF>(key) >= b_low - std::numeric_limits<phi_type>::epsilon()) {
            // Here, change if-condition accordingly to where you would like to have the source
            if(g_dist.template get<PHI_SDF>(key) < 0.15*phi_max && x < x_max){
                g_dist.template get<K_SOURCE>(key) = k_source;
                g_dist.template get<K_SINK>(key) = 0;
            }
            else if (g_dist.template get<PHI_SDF>(key) < 0.15*phi_max && x > x_max) {
                g_dist.template get<K_SOURCE>(key) = 0;
                g_dist.template get<K_SINK>(key) = k_sink;
            }
            else {
                g_dist.template get<K_SOURCE>(key) = 0;
                g_dist.template get<K_SINK>(key) = 0;
            }
        }
            
		++dom;
	}


	g_dist.write(path_output + "grid_initial", FORMAT_BINARY); // Save initial grid
	

    
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Diffusion using a forward-time central-space scheme
	double t = 0;
	int iter = 0; // initial iteration
	int max_iter = 1e4; // max iteration
	int interval_write = (int)(max_iter / 100); // set how many frames should be saved as vtk
	
	double p_volume  = dx * dy;
	double m_initial = sum_prop_over_region<CONC_N, PHI_SDF>(g_dist, b_low) * p_volume; // Initial total mass
	std::cout << "m_initial : " << m_initial << std::endl;

	if (m_initial == 0){
        m_initial = 0.01;
    }


	while(iter < max_iter)
	{
		// Impose no-flux BCs at the disk interface
		auto dom4 = g_dist.getDomainIterator();
		while(dom4.isNext()) {
			auto key = dom4.get();
			if(g_dist.template getProp<PHI_SDF>(key) >= b_low - std::numeric_limits<phi_type>::epsilon()) {
				for(int d = 0; d < dims; ++d) {
					if(g_dist.template get<PHI_SDF>(key.move(d, 1)) < b_low + std::numeric_limits<double>::epsilon()) {
						g_dist.template get<CONC_N>(key.move(d, 1)) = g_dist.template get<CONC_N>(key);
					}
					
					if(g_dist.template get<PHI_SDF>(key.move(d, -1)) < b_low + std::numeric_limits<double>::epsilon()) {
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
            if (g_dist.template getProp<PHI_SDF>(key) >= b_low - std::numeric_limits<phi_type>::epsilon())
            {
                g_dist.template get<CONC_NPLUS1>(key) = g_dist.template get<CONC_N>(key) + D * dt * g_dist.template get<CONC_LAP>(key) 
                + dt*g_dist.template get<K_SOURCE>(key) - dt*g_dist.template get<K_SINK>(key)*g_dist.template get<CONC_N>(key);            
            }
            
            
            ++dom3;
        }
		
		
		// Write grid to vtk
		if (iter % interval_write == 0)
		{
			g_dist.write_frame(path_output + "/grid_diffuse_withNoFlux", iter, FORMAT_BINARY);
			std::cout << "Diffusion time :" << t << std::endl;
			
			// Monitor mass to check conservation in case of no reaction
			// Write mass to csv
			monitor_absolute_mass_over_region<CONC_N, PHI_SDF>(g_dist, b_low, m_initial, p_volume, t, iter, path_output, "mass.csv");
		}
		
		// Update CONC_N
		copy_gridTogrid<CONC_NPLUS1, CONC_N>(g_dist, g_dist);
		
		iter += 1;
		t += dt;
	}
}
