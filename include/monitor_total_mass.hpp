//
// Created by jstark on 20.05.22.
//

#ifndef DIFFUSION_MONITOR_TOTAL_MASS_HPP
#define DIFFUSION_MONITOR_TOTAL_MASS_HPP


/**@brief Sums up property values over grid nodes that lie within Phi > b_low.
 *
 * @tparam Prop Index of property whose values will be summed up over grid nodes that lie within Phi > b_low.
 * @tparam grid_type Template type of input grid.
 * @param grid Input grid.
 * @return Sum of values stored in Prop over grid nodes that lie within Phi > b_low
 */
template <size_t Prop, size_t Phi, typename grid_type, typename phi_type>
auto sum_prop_over_region(grid_type & grid, const phi_type b_low)
{
	auto sum = 0.0;
	auto dom = grid.getDomainIterator();
	while(dom.isNext())
	{
		auto key = dom.get();
		if(grid.template getProp<Phi>(key) >= b_low - std::numeric_limits<phi_type>::epsilon())
		{
			sum += grid.template getProp<Prop>(key);
		}
		++dom;
	}
	auto &v_cl = create_vcluster();
	v_cl.sum(sum);
	v_cl.execute();
	return sum;
}


/**@brief Writing out the total mass within region defined by phi normalized by initial mass to a csv file.
 *
 * @tparam Conc Index of property containing the concentration.
 * @tparam Phi Index of property containing the signed distance function.
 * @tparam grid_type Template type of input grid.
 * @tparam phi_type Template type of minimum phi (b_low) considered as region boundary
 * @tparam T Template type of time and node volume.
 * @tparam T_mass Template type of mass.
 * @param grid Input grid.
 * @param b_low Minimum phi (b_low) considered as region boundary
 * @param m_initial Initial total mass (for comparison to monitor in/decrease).
 * @param p_volume Volume of grid node.
 * @param t Time.
 * @param i Iteration.
 * @param path_output Output path where csv file should be saved.
 * @param filename Name of csv file.
 */
template <size_t Conc, size_t Phi, typename grid_type, typename phi_type, typename T, typename T_mass>
void monitor_absolute_mass_over_region(grid_type & grid, const phi_type b_low, const T_mass m_initial, const T p_volume, const T t, const size_t i,
const std::string & path_output, const std::string & filename="normalized_total_mass.csv")
{
	auto &v_cl = create_vcluster();

	if (m_initial == 0)
	{
		if (v_cl.rank() == 0) std::cout << "m_initial is zero! Normalizing the total mass with the initial mass will "
		                                   "not work. Aborting..." << std::endl;
		abort();
	}

	T m_total = sum_prop_over_region<Conc, Phi>(grid, b_low) * p_volume;

	if (v_cl.rank() == 0)
	{
		std::string outpath = path_output + "/" + filename;
		create_file_if_not_exist(outpath);
		std::ofstream outfile;
		outfile.open(outpath, std::ios_base::app); // append instead of overwrite

		if (i == 0)
		{
			outfile   << "Time, Total mass" << std::endl;
			std::cout << "Time, Total mass" << std::endl;
		}

		outfile
				<< to_string_with_precision(t, 6) << ","
				<< to_string_with_precision(m_total, 6) << ","
				<< std::endl;
		outfile.close();

		std::cout
				<< to_string_with_precision(t, 6) << ","
				<< to_string_with_precision(m_total, 6) << ","
				<< std::endl;
	}
}
#endif //DIFFUSION_MONITOR_TOTAL_MASS_HPP