#ifndef NUMCP_UTIL
#define NUMCP_UTIL

#include <vector>
#include <cmath>
#include <cblas.h>
#include <array.hpp>

namespace numcp {

//template<typename T_> Array<T_> dot (Array<T_>& lhs, Array<T_>& rhs); 
Array<double> dot (const Array<double>& lhs, const Array<double>& rhs); // double support only
template<typename T_> T_ norm (const Array<T_>& ary); 
template<typename T_> T_ sum (const Array<T_>& ary); 
template<typename T_> T_ mean (const Array<T_>& ary); 
template<typename T_> T_ max (const Array<T_>& ary); 
template<typename T_> T_ min (const Array<T_>& ary); 
template<typename T_> Array<T_> ones (const std::vector<int> shape); 
template<typename T_> Array<T_> zeros (const std::vector<int> shape); 


Array<double> dot (const Array<double>& lhs, const Array<double>& rhs)
{
	CHECK_EQ(lhs.ndim(), rhs.ndim());
	CHECK_LT(lhs.ndim(), 3);
	CHECK_LT(rhs.ndim(), 3);

	Array<double> ret;
	// 1D vector dot
	if (lhs.ndim()==1 && rhs.ndim()==1)
	{
		// shape check
		CHECK_EQ(lhs.size(), rhs.size());
		ret = Array<double>({1});
		
		// using cblas ddot
		ret.data[0] = cblas_ddot (lhs.size(),
								  lhs.data,
								  1,
								  rhs.data,
								  1);
	}
	// 2D matrix dot
	else if (lhs.ndim()==2 && rhs.ndim()==2)
	{
		// shape check
		CHECK_EQ(lhs.shape()[1], rhs.shape()[0]);
		ret = Array<double>({lhs.shape()[0], rhs.shape()[1]});
		
		// using cblas dgemm
		cblas_dgemm (CblasColMajor,
					 CblasNoTrans,
					 CblasNoTrans,
					 lhs.shape()[0],
					 rhs.shape()[1],
					 lhs.shape()[1],
					 1.,
					 lhs.data,
					 lhs.shape()[0],
					 rhs.data,
					 rhs.shape()[0],
					 0.,
					 ret.data,
					 ret.shape()[0]);
	}
	return ret;
}


template<typename T_>
T_ norm (const Array<T_>& ary)
{
	T_ ret = ary.data[0] * ary.data[0];
	for (int i=1; i<ary.size(); ++i) ret += ary.data[i] * ary.data[i];
	return std::sqrt(ret);
}


template<typename T_>
T_ sum (const Array<T_>& ary)
{
	T_ ret = ary.data[0];
	for (int i=1; i<ary.size(); ++i) ret += ary.data[i];
	return ret;
}



template<typename T_>
T_ mean (const Array<T_>& ary)
{
	return sum(ary) / ary.size(); 
}


template<typename T_>
T_ max (const Array<T_>& ary)
{
	T_ ret = ary.data[0];
	for (int i=1; i<ary.size(); ++i) ret = (ret > ary.data[i]) ? ret : ary.data[i];
	return ret;
}


template<typename T_>
T_ min (const Array<T_>& ary)
{
	T_ ret = ary.data[0];
	for (int i=1; i<ary.size(); ++i) ret = (ret < ary.data[i]) ? ret : ary.data[i];
	return ret;
}


template<typename T_>
Array<T_> ones (const std::vector<int> shape)
{
	return Array<T_>(shape, (T_)1);
}


template<typename T_>
Array<T_> zeros (const std::vector<int> shape)
{
	return Array<T_>(shape, (T_)0);
}

} // namespace numcp

#endif
