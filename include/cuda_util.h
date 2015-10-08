#ifndef NUMCP_CUDA_UTIL
#define NUMCP_CUDA_UTIL

#include "darray.hpp"

namespace numcp {
	Darray<double> cudot(const Darray<double>& lhs, const Darray<double>& rhs); 
	Darray<float> cudot(const Darray<float>& lhs, const Darray<float>& rhs); 
	double cunorm2 (const Darray<double>& ary); 
	float cunorm2 (const Darray<float>& ary); 
}
#endif
