//
// Created by jstark on 20.05.22.
//

#ifndef TIMESTEPS_STABILITY_HPP
#define TIMESTEPS_STABILITY_HPP

/**@brief Get timestep that fulfills stability criterion nD-diffusion using a FTCS scheme
 *
 * @tparam grid_type Template type of input g_sparse.
 * @tparam T Template type of diffusion coefficient and timestep.
 * @param grid Input grid of type grid_type.
 * @param k_max Max. diffusion coefficient in the domain.
 * @return T type timestep.
 */
template <typename grid_type, typename T>
typename grid_type::stype get_diffusion_time_step(grid_type & grid, T k_max)
{
	typename grid_type::stype sum = 0;
	for (int d = 0; d < grid_type::dims; ++d)
	{
		sum += 1 / (grid.spacing(d) * grid.spacing(d));
	}
	
	return 1 / (4 * k_max * sum);
}

/**@brief Computes the time step size fulfilling CFL condition according to https://www.cfd-online
 * .com/Wiki/Courant–Friedrichs–Lewy_condition for arbitrary dimensionality.
 *
 * @tparam grid_type Template type of the input grid.
 * @param grid Input OpenFPM grid.
 * @param u Array of size grid_type::dims containing the velocity in each dimension.
 * @param C Courant number.
 * @return Time step.
 */
template <typename grid_type>
typename grid_type::stype get_advection_time_step_cfl(grid_type & grid, typename grid_type::stype u [grid_type::dims], 
											   typename grid_type::stype C)
{
	typename grid_type::stype sum = 0;
	for (size_t d = 0; d < grid_type::dims; d++)
	{
		sum += u[d] / grid.spacing(d);
	}
	return C / sum;
}

/**@brief Computes the time step size fulfilling CFL condition according to https://www.cfd-online
 * .com/Wiki/Courant–Friedrichs–Lewy_condition for arbitrary dimensionality.
 *
 * @tparam grid_type Template type of the input grid.
 * @param grid Input OpenFPM grid.
 * @param u Velocity of propagating wave if isotropic for each direction.
 * @param C Courant number.
 * @return Time step.
 */
template <typename grid_type>
typename grid_type::stype get_advection_time_step_cfl(grid_type & grid, typename grid_type::stype u, typename grid_type::stype C)
{
	typename grid_type::stype sum = 0;
	for (size_t d = 0; d < grid_type::dims; d++)
	{
		sum += u / grid.spacing(d);
	}
	return C / sum;
}

#endif //TIMESTEPS_STABILITY_HPP