// source /Users/home/juwailes/openfpm_vars
// cmake ../. -DCMAKE_BUILD_TYPE=Release;


#include <iostream>
#include "util/PathsAndFiles.hpp"
#include <fstream>
#include <cstdio>
#include <ctime>

// these are available in openfpm_data and openfpm_install_master
// Analytical SDF to define the disk-shaped diffusion domain
#include "level_set/redistancing_Sussman/AnalyticalSDF.hpp" 
#include "level_set/redistancing_Sussman/HelpFunctionsForGrid.hpp"

// where are these?
// not in dependencies or install
#include "FD_laplacian.hpp"
#include "Gaussian.hpp"
#include "timesteps_stability.hpp"
#include "monitor_total_mass.hpp"


// Grid parameters setup
const size_t dims = 1;
constexpr size_t x = 0;
const double leftBoundary = 0.0, rightBoundary = 4.0;
const double centerOfSDF = 2, radiusOfSDFInterval = 1;
const double b_low = 0;

// Properties numbering and grouping
constexpr size_t PHI_SDF = 0, CONC_N = 1, CONC_NPLUS1 = 2, CONC_LAP = 3, DIFFUSION_COEFFICIENT = 4, K_SOURCE = 5, K_SINK = 6;
typedef aggregate<double, double, double, double, double, double, double> properties;

// Reaction-Diffusion Parameters
const double D = 0.5;
const double source = 1;
const double sink = 1;


int main(int argc, char* argv[])
{
	// Time counter 
	std::clock_t start;
    double duration;
    start = std::clock();

	// Initialize library
	openfpm_init(&argc, &argv);
	auto & v_cl = create_vcluster();

	std::string cwd = get_cwd();
	const std::string path_output = cwd + "/output_diffusion/";
	create_directory_if_not_exist(path_output);

	// Grid definition
	const size_t N = 128;
	const size_t size[dims] = {N};

	Box<dims, double> box({leftBoundary}, {rightBoundary});
	Ghost<dims, long int> ghost(1);
	grid_dist_id<dims, double, properties> g_dist(size, box, ghost);	

	// Assigning the properties to the grid with initial values
	g_dist.setPropNames({"PHI_SDF", "CONC_N", "CONC_NPLUS1", "CONC_LAP", "DIFFUSION_COEFFICIENT", "K_SOURCE", "K_SINK"});
	init_grid_and_ghost<CONC_N>(g_dist, 0);
	init_grid_and_ghost<CONC_NPLUS1>(g_dist, 0);
	init_grid_and_ghost<PHI_SDF>(g_dist, -1);

	
	// Calculate the SDF for all grid points
	typedef grid_dist_id<dims, double, properties> grid_type;
	double xMaxPhi = 0, MaxPhi = 0; 
	auto dom = g_dist.getDomainIterator();
	while(dom.isNext())
	{
		auto key = dom.get();

		Point<grid_type::dims, typename grid_type::stype> coords = g_dist.getPos(key); 
		auto x = coords.get(0);

		double length = abs(x - centerOfSDF);
		double Phi = radiusOfSDFInterval - length;
		g_dist.template get<PHI_SDF>(key) = Phi;
		
		// Store the location and value of largest SDF-value 
		if(Phi > MaxPhi)
		{
			MaxPhi = Phi;
			xMaxPhi = x;
		}

		++dom;
	}
	std::cout << "Maximum SDF is at " << xMaxPhi << ", with a value of " << MaxPhi << std::endl;


	// Placing of sources and sinks
	auto dom2 = g_dist.getDomainIterator();
	while(dom2.isNext())
	{
		auto key = dom2.get();

		Point<grid_type::dims, typename grid_type::stype> coords = g_dist.getPos(key); 
		auto x = coords.get(0);

		if (g_dist.template getProp<PHI_SDF>(key) >= b_low - std::numeric_limits<double>::epsilon())
		{
			if(g_dist.template getProp<PHI_SDF>(key) < 0.2*radiusOfSDFInterval && x < centerOfSDF)
			{
				g_dist.template getProp<K_SOURCE>(key) = source;
				g_dist.template getProp<K_SINK>(key) = 0;
			}
			else if(g_dist.template getProp<PHI_SDF>(key) < 0.2*radiusOfSDFInterval && x > centerOfSDF)
			{
				g_dist.template getProp<K_SOURCE>(key) = 0;
				g_dist.template getProp<K_SINK>(key) = sink;
			}
			else
			{
				g_dist.template getProp<K_SOURCE>(key) = 0;
				g_dist.template getProp<K_SINK>(key) = 0;
			}
		} 

		++dom2;
	} 


    // stability condition
	auto dx = g_dist.spacing(x);
	auto dt = get_diffusion_time_step(g_dist, D);
	std::cout << "dx = " << dx << ", dt = " << dt << std::endl;


	////////////////////////////////////////////////////////////////////////////////////////////////////////
	//												DE Solving                                            //
	////////////////////////////////////////////////////////////////////////////////////////////////////////


	// Setting the number of solution time-steps and vtk frames 
	int currenterIteration = 0;
	const int maxIteration = 10000;
	const int numberOfFrames = maxIteration/100; 

	double t = 0;
	while(t < 100)
	{
		
		// Imposing no-flux BC's 
		auto dom3 = g_dist.getDomainIterator();
		while(dom3.isNext())
		{
			auto key = dom3.get();

			if(g_dist.template getProp<PHI_SDF>(key) >= b_low - std::numeric_limits<double>::epsilon())
			{
				for(int d = 0; d < dims; ++d)
				{
					if(g_dist.template get<PHI_SDF>(key.move(d, 1)) < b_low + std::numeric_limits<double>::epsilon())
					{g_dist.template get<CONC_N>(key.move(d, 1)) = g_dist.template get<CONC_N>(key);}
					if(g_dist.template get<PHI_SDF>(key.move(d, -1)) < b_low + std::numeric_limits<double>::epsilon())
					{g_dist.template get<CONC_N>(key.move(d, -1)) = g_dist.template get<CONC_N>(key);}
				}
			}

			++dom3;
		}

		// Laplacian
		get_laplacian_grid<CONC_N, CONC_LAP>(g_dist);

		// Solving the DE
		auto dom4 = g_dist.getDomainIterator();
		while(dom4.isNext())
		{
			auto key = dom4.get();

			if(g_dist.template getProp<PHI_SDF>(key) >= b_low - std::numeric_limits<double>::epsilon()) 
			{
				g_dist.template get<CONC_NPLUS1>(key) = g_dist.template get<CONC_N>(key) + D * dt * g_dist.template get<CONC_LAP>(key) 
                + dt*g_dist.template get<K_SOURCE>(key) - dt*g_dist.template get<K_SINK>(key)*g_dist.template get<CONC_N>(key);
			}

			++dom4;
		}


		// Produce VTK files
		/*if (currenterIteration % numberOfFrames == 0)
		{
			g_dist.write_frame(path_output + "/grid_diffuse_withNoFlux", currenterIteration, FORMAT_BINARY);
			std::cout << "Diffusion time :" << t << std::endl;
		}*/
		
		// Update CONC_N
		copy_gridTogrid<CONC_NPLUS1, CONC_N>(g_dist, g_dist);
		
		currenterIteration += 1;
		t += dt;
	}


	// write to a CSV file
	std::string outpath = path_output + "/" + "Concentration.csv";
	create_file_if_not_exist(outpath);
	std::ofstream excelFile;
	excelFile.open(outpath);
	
	int i = 0;
	double concentration = 0;
	auto dom5 = g_dist.getDomainIterator();
	while(dom5.isNext())
	{
		auto key = dom5.get();
		
		if (i == 0){excelFile   << "x-coordinate,concentration" << std::endl;}
		++i;
		
		Point<grid_type::dims, typename grid_type::stype> coords = g_dist.getPos(key); 
		auto x = coords.get(0);
		concentration = g_dist.template get<CONC_N>(key);
		excelFile << x << "," << concentration << std::endl;

		++dom5;
	}
	excelFile.close();

	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Operation took "<< duration << "seconds" << std::endl;
}