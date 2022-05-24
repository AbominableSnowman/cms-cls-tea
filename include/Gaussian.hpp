//
// Created by jstark on 02.05.22.
//

#ifndef POROUS_RAD_GAUSSIAN_HPP
#define POROUS_RAD_GAUSSIAN_HPP

#include "cmath"

template<typename T>
static T hermite_polynomial(T x, T sigma, int order)
{
	T h;
	switch(order)
	{
		case 0:
			h = 1;
			break;
		case 1:
			h = -x / (sigma * sigma);
			break;
		case 2:
			h = (x*x - sigma*sigma) / (sigma*sigma*sigma*sigma);
			break;
		default:
			std::cout << "Only gaussian derivatives of order 0, 1, and 2 implemented. Aborting..." << std::endl;
			abort();
	}
	return h;
}

template <typename point_type, typename T>
T gaussian_1D(const point_type & x, const T mu, const T sigma)
{
	const T pi = 3.14159265358979323846;
	const T sqrt2pi = sqrt(2*pi);
	
	T sum = (x - mu) * (x - mu) / (sigma*sigma);;
	T normalization_factor = 1 / (sqrt2pi * sigma);
	
	return normalization_factor * exp(-0.5 * sum);
}

template <typename point_type, typename T>
T gaussian_1D(const point_type & x, const T mu, const T sigma, const T normalization_factor)
{
	T sum = (x - mu) * (x - mu) / (sigma*sigma);;
	return normalization_factor * exp(-0.5 * sum);
}

template <typename point_type, typename T>
T gaussian(const point_type & x, const T mu[point_type::dims], const T sigma[point_type::dims])
{
	T g = 1;
	for(int d=0; d<point_type::dims; d++)
	{
		g *= gaussian_1D(x.get(d), mu[d], sigma[d]);
	}
	return g;
}

template <typename point_type, typename T>
T gaussian(const point_type & x,
		   const T mu[point_type::dims],
		   const T sigma[point_type::dims],
		   const T normalization_factor[point_type::dims])
{
	T g = 1;
	for(int d=0; d<point_type::dims; d++)
	{
		g *= gaussian_1D(x.get(d), mu[d], sigma[d], normalization_factor[d]);
	}
	return g;
}
#endif //POROUS_RAD_GAUSSIAN_HPP
