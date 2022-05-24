//
// Created by jstark on 20.05.22.
//

#ifndef DIFFUSION_TIMESTEPS_STABILITY_HPP
#define DIFFUSION_TIMESTEPS_STABILITY_HPP

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

#endif //DIFFUSION_TIMESTEPS_STABILITY_HPP
