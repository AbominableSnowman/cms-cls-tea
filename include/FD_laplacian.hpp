//
// Created by jstark on 20.05.22.
//

#ifndef DIFFUSION_FD_LAPLACIAN_HPP
#define DIFFUSION_FD_LAPLACIAN_HPP

/**@brief Computes the 2nd order central finite difference of a scalar field on the current grid node.
 *
 * @tparam FIELD Size_t index of property for which the laplacian should be computed.
 * @tparam gridtype Type of input grid.
 * @tparam keytype Type of key variable.
 * @param grid Grid, on which the laplacian should be computed.
 * @param key Key that contains the index of the current grid node.
 * @param d Variable (size_t) that contains the dimension.
 * @return Partial 2nd order central finite difference for dimension d of FIELD on the current node with index key.
 */
template <size_t FIELD, typename gridtype, typename keytype>
auto FD_central_2nd_derivative(gridtype & grid, keytype & key, size_t d)
{
	return (grid.template get<FIELD>(key.move(d, 1)) // moves one grid point to the right
			+ grid.template get<FIELD>(key.move(d, -1)) // moves one grid point to the left
			 - 2 * grid.template get<FIELD>(key))
			/ (grid.getSpacing()[d] * grid.getSpacing()[d]);
}

/**@brief Computes the laplacian using central finite difference of a scalar field on the full grid.
 *
 * @tparam FIELD Size_t index of input property for which the laplacian should be computed (scalar field).
 * @tparam LAPLACIAN Size_t index of output property (scalar field).
 * @tparam gridtype Type of input grid.
 * @param grid Grid, on which the laplacian should be computed.
 */
template <size_t FIELD, size_t LAPLACIAN, typename gridtype>
void get_laplacian_grid(gridtype & grid)
{
	grid.template ghost_get<FIELD>(KEEP_PROPERTIES);
	auto dom = grid.getDomainIterator();
	
	while (dom.isNext())
	{
		auto key = dom.get();
		grid.template get<LAPLACIAN>(key) = 0;
		// Get laplacian of FIELD by running 2nd order finite difference over all dimensions and summing up the
		// partial derivatives
		for(size_t d = 0; d < gridtype::dims; d++)
		{
			grid.template get<LAPLACIAN>(key) += FD_central_2nd_derivative<FIELD>(grid, key, d);
		}
				
		++dom;
	}
	
}


#endif //DIFFUSION_FD_LAPLACIAN_HPP
