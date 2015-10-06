#ifndef NUMCP_CUDA
#define NUMCP_CUDA

#include "darray.hpp"

namespace numcp {

template<typename T_> Darray<T_> cudot(const Darray<T_>& lhs, const Darray<T_>& rhs); 
template<typename T_> T_ cunorm (const Darray<T_>& ary); 
template<typename T_> T_ cusum (const Darray<T_>& ary); 

} // namespace numcp
#endif
