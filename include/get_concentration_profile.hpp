//
// Created by jstark on 20.05.22.
//

#ifndef EVALUATION_GET_CONCENTRATION_PROFILE_HPP
#define EVALUATION_GET_CONCENTRATION_PROFILE_HPP


#include "Grid/grid_dist_id.hpp"
#include "VCluster/VCluster.hpp"

template <typename point_type, typename y_margin_type>
auto distance_from_margin(point_type & coord, y_margin_type y_margin)
{
	return(coord.get(1) - y_margin);
}

template <size_t PROP, typename grid_type, typename ny_type, typename startstop_type,
		typename Ty, typename scaling_type, typename distance_type, typename mass_type>
void get_1D_profile_from_2D(grid_type & grid,
                            const ny_type Ny,
                            const startstop_type x_start,
                            const startstop_type x_stop,
                            const Ty y_margin,
                            const scaling_type scaling,
                            openfpm::vector<distance_type> & v_distance_from_margin,
                            openfpm::vector<mass_type> & v_mass_slice)
{
	size_t total_points = 0;
	for (startstop_type j = 0; j < Ny; ++j)
	{
		auto dom = grid.getSubDomainIterator({x_start, j}, {x_stop, j});
		Point<grid_type::dims, typename grid_type::stype> coord = grid.getPos(dom.get());

		mass_type sum = 0.0;
		while (dom.isNext())
		{
			auto key = dom.get();
			sum += grid.template getProp<PROP>(key) * (mass_type) scaling;
			total_points += 1;
			++dom;
		}
		v_mass_slice.get(j) = sum;
		v_distance_from_margin.get(j) = (distance_from_margin(coord, y_margin));
#ifdef SE_CLASS1
		std::cout << "Total number of points considered in summation is " << total_points << std::endl;
		std::cout << "Grid size is " << grid.size() << std::endl;
		std::cout << "---------------------------------------------------------------------------" << std::endl;
		std::cout << "v_mass_slice.size() = " << v_mass_slice.size() << std::endl;
		std::cout << "v_distance_from_margin.size() = " << v_distance_from_margin.size() << std::endl;
		std::cout << "Both vectors should have length = " << Ny << std::endl;
		if(Ny == grid.size(1))
		{
			assert(total_points == grid.size());
			assert(v_mass_slice.size() == Ny);
			assert(v_distance_from_margin.size() == Ny);
		}
		else
		{
			std::cout << "WARNING: Ny is not equal to size of grid in y direction!" << std::endl;
		}
#endif // SE_CLASS1
	}
}

template <size_t PROP, typename grid_type, typename ny_type, typename startstop_type,
		typename Ty, typename scaling_type, typename distance_type, typename mass_type>
void get_1D_profile_from_3D(grid_type & grid,
                            const ny_type Ny,
                            const startstop_type x_start,
                            const startstop_type x_stop,
                            const startstop_type z_start,
                            const startstop_type z_stop,
                            const Ty y_margin,
                            const scaling_type scaling,
                            openfpm::vector<distance_type> & v_distance_from_margin,
                            openfpm::vector<mass_type> & v_mass_slice)
{
	size_t total_points = 0;
	for (startstop_type j = 0; j < Ny; ++j)
	{
		auto dom = grid.getSubDomainIterator({x_start, j, z_start}, {x_stop, j, z_stop});
		Point<grid_type::dims, typename grid_type::stype> coord = grid.getPos(dom.get());

		mass_type sum = 0.0;
		while (dom.isNext())
		{
			auto key = dom.get();
			sum += grid.template getProp<PROP>(key) * (mass_type) scaling;
			total_points += 1;
			++dom;
		}
		v_mass_slice.get(j) = sum;
		v_distance_from_margin.get(j) = (distance_from_margin(coord, y_margin));
	}
#ifdef SE_CLASS1
	std::cout << "Total number of points considered in summation is " << total_points << std::endl;
	std::cout << "Grid size is " << grid.size() << std::endl;
	std::cout << "---------------------------------------------------------------------------" << std::endl;
	std::cout << "v_mass_slice.size() = " << v_mass_slice.size() << std::endl;
	std::cout << "v_distance_from_margin.size() = " << v_distance_from_margin.size() << std::endl;
	std::cout << "Both vectors should have length = " << Ny << std::endl;
	if(Ny == grid.size(1))
	{
		assert(total_points == grid.size());
		assert(v_mass_slice.size() == Ny);
		assert(v_distance_from_margin.size() == Ny);
	}
	else
	{
		std::cout << "WARNING: Ny is not equal to size of grid in y direction!" << std::endl;
	}
#endif // SE_CLASS1
}

template <typename v1_type, typename v2_toReduce_type>
void reduction_and_write_vectors_to_csv(const openfpm::vector<v1_type> & v1,
                                        openfpm::vector<v2_toReduce_type> & v2_toReduce,
                                        const std::string & path_output,
                                        const std::string & filename="reducedVector.csv",
                                        const int precision=6)
{
	assert(v1.size() == v2_toReduce.size());
	std::string path_output_summed_mass = path_output + "/" + filename;

	auto & v_cl = create_vcluster();
	if (v_cl.rank() == 0)
	{
		create_file_if_not_exist(path_output_summed_mass);
	}

	for(size_t elem = 0; elem < v1.size(); ++elem)
	{
		v_cl.sum(v2_toReduce.get(elem));
		v_cl.execute();
//
		// Save to csv file for plotting
		if (v_cl.rank() == 0)
		{
			std::ofstream file_out;
			file_out.open(path_output_summed_mass, std::ios_base::app); // append instead of overwrite

			file_out << to_string_with_precision(v1.get(elem), precision)
					<< ',' << to_string_with_precision(v2_toReduce.get(elem), precision) << std::endl;

			file_out.close();
		}
	}
}

template<size_t PHI_SDF, size_t PROP_OUT, typename grid_type, typename phi_type, typename T>
void load_vector_onto_2D_grid_singleProcessor(grid_type & grid,
                                              phi_type b_low,
                                              phi_type b_up,
                                              openfpm::vector<T> v_lin,
                                              size_t m,
                                              size_t n,
                                              size_t col_to_load=1)
{
	constexpr size_t x = 0;
	constexpr size_t y = 1;

	// Asserting that we are running on a single processor only because this implementation will not (yet) work in
	// parallel
	auto & v_cl = create_vcluster();
	assert(v_cl.size() == 1);
	// Asserting that number of rows equals the grid-size along y
	assert(m == grid.size(y));

	typename grid_type::stype p_volume = grid.spacing(x) * grid.spacing(y);
	const int x_start = 0, x_stop = grid.size(x);
	for(int j = 0; j < grid.size(y); ++j)
	{
		// Get the number of grid points along the dorsal-ventral axis that are located inside the diffusion space
		size_t points_along_dorsal_ventral = 0;
		auto dom = grid.getSubDomainIterator({x_start, j}, {x_stop, j});
		while (dom.isNext())
		{
			auto key = dom.get();
			if (is_diffusionSpace(grid.template get<PHI_SDF>(key), b_low, b_up))
			{
				points_along_dorsal_ventral += 1;
			}
			++dom;
		}

		// Divide the total amount stored for a specific distance to the margin by the number of grid nodes along
		// the dorsal-ventral axis
		T divided_amount =
				v_lin.get(j * n + col_to_load) /
						((T) points_along_dorsal_ventral + std::numeric_limits<T>::epsilon());
//		std::cout << "j = " << j << ", divided conc = " << divided_amount << std::endl;

		// Write the divided amount onto every grid node inside diffusion space along the dorsal-ventral axis
		auto dom2 = grid.getSubDomainIterator({x_start, j}, {x_stop, j});
		while (dom2.isNext())
		{
			auto key = dom2.get();
			if (is_diffusionSpace(grid.template get<PHI_SDF>(key), b_low, b_up))
			{
				grid.template insertFlush<PROP_OUT>(key) = divided_amount;
			}
			++dom2;
		}
	}
}

template<size_t PHI_SDF, size_t PROP_OUT, typename grid_type, typename phi_type, typename T>
void load_vector_onto_3D_grid_singleProcessor(grid_type & grid,
                                              phi_type b_low,
                                              phi_type b_up,
                                              openfpm::vector<T> v_lin,
                                              size_t m,
                                              size_t n,
                                              size_t col_to_load=1)
{
	constexpr size_t x = 0;
	constexpr size_t y = 1;
	constexpr size_t z = 2;

	// Asserting that we are running on a single processor only because this implementation will not (yet) work in
	// parallel
	auto & v_cl = create_vcluster();
	assert(v_cl.size() == 1);
	// Asserting that number of rows equals the grid-size along y
	assert(m == grid.size(y));

	typename grid_type::stype p_volume = grid.spacing(x) * grid.spacing(y);
	const int x_start = 0, x_stop = grid.size(x);
	const int z_start = 0, z_stop = grid.size(z);
	for(int j = 0; j < grid.size(y); ++j)
	{
		// Get the number of grid points along the dorsal-ventral axis that are located inside the diffusion space
		size_t points_along_dorsal_ventral = 0;
		auto dom = grid.getSubDomainIterator({x_start, j, z_start}, {x_stop, j, z_stop});
		while (dom.isNext())
		{
			auto key = dom.get();
			if (is_diffusionSpace(grid.template get<PHI_SDF>(key), b_low, b_up))
			{
				points_along_dorsal_ventral += 1;
			}
			++dom;
		}

		// Divide the total amount stored for a specific distance to the margin by the number of grid nodes along
		// the dorsal-ventral axis
		T divided_amount =
				v_lin.get(j * n + col_to_load) /
						((T) points_along_dorsal_ventral + std::numeric_limits<T>::epsilon());

		// Write the divided amount onto every grid node inside diffusion space along the dorsal-ventral axis
		auto dom2 = grid.getSubDomainIterator({x_start, j, z_start}, {x_stop, j, z_stop});
		while (dom2.isNext())
		{
			auto key = dom2.get();
			if (is_diffusionSpace(grid.template get<PHI_SDF>(key), b_low, b_up))
			{
				grid.template insertFlush<PROP_OUT>(key) = divided_amount;
			}
			++dom2;
		}
	}
}

template<size_t PHI_SDF, size_t PROP_OUT, typename grid_type, typename phi_type, typename T>
void get_field_from_vector(grid_type & grid,
                           phi_type b_low,
                           phi_type b_up,
                           openfpm::vector<T> v_lin,
                           size_t m,
                           size_t n,
                           size_t col_to_load=1)
{
	// Asserting that we are running on a single processor only because this implementation will not (yet) work in
	// parallel
	auto &v_cl = create_vcluster();
	assert(v_cl.size() == 1);

	if (grid_type::dims == 2)
	{
		load_vector_onto_2D_grid_singleProcessor<PHI_SDF, PROP_OUT>(grid, b_low, b_up, v_lin, m, n, col_to_load);
	}

	if (grid_type::dims == 3)
	{
		load_vector_onto_3D_grid_singleProcessor<PHI_SDF, PROP_OUT>(grid, b_low, b_up, v_lin, m, n, col_to_load);
	}

	else
	{
		std::cout << "-------------------------------------------------------------------------" << std::endl;
		std::cout << "Grid has dimension " << grid_type::dims
				<< ", but dimension must be 2 or 3 for loading the vector. Aborting..." << std::endl;
		abort();
		std::cout << "-------------------------------------------------------------------------" << std::endl;
	}

}


#endif //EVALUATION_GET_CONCENTRATION_PROFILE_HPP
