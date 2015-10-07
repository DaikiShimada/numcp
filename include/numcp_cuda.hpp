#ifndef NUMCP_CUDA
#define NUMCP_CUDA

#include "numcp_cblas_helper.h"
#include "darray.hpp"
#include "util.hpp"

namespace numcp {

Darray<double> cudot(const Darray<double>& lhs, const Darray<double>& rhs); 
Darray<float> cudot(const Darray<float>& lhs, const Darray<float>& rhs); 
double cunorm2 (const Darray<double>& ary); 
float cunorm2 (const Darray<float>& ary); 


Darray<double> cudot (const Darray<double>& lhs, const Darray<double>& rhs)
{
	// context check
	CHECK_EQ(lhs.getDeviceManager().getDeviceID(), rhs.getDeviceManager().getDeviceID());
	
	CHECK_EQ(lhs.ndim(), rhs.ndim());
	CHECK_LT(lhs.ndim(), 3);
	CHECK_LT(rhs.ndim(), 3);

	Darray<double> ret;

	if (lhs.ndim()==1 && rhs.ndim()==1)
	{
		// shape check
		CHECK_EQ(lhs.size(), rhs.size());
		ret = Darray<double>(lhs.getDeviceManager(), {1});
		
		// using cublas ddot
		lhs.deviceSet();
		cublasDdot (DeviceManager::handle,
				    lhs.size(),
				    lhs.data,
				    1,
				    rhs.data,
				    1,
				    ret.data);
	}
	// 2D matrix dot
	else if (lhs.ndim()==2 && rhs.ndim()==2)
	{
		// shape check
		CHECK_EQ(lhs.shape()[1], rhs.shape()[0]);
		ret = Darray<double>(lhs.getDeviceManager(), {lhs.shape()[0], rhs.shape()[1]});
		
		// using cblas dgemm
		lhs.deviceSet();
		const double alpha = 1.;
		const double beta = 0.;
		CUBLAS_SAFE_CALL(
		cublasDgemm (DeviceManager::handle,
					CUBLAS_OP_N,
					CUBLAS_OP_N,
					lhs.shape()[0],
					rhs.shape()[1],
					lhs.shape()[1],
					&alpha,
					lhs.dev_data,
					lhs.shape()[0],
					rhs.dev_data,
					rhs.shape()[0],
					&beta,
					ret.dev_data,
					ret.shape()[0])
		);
	}
	return ret;
}


Darray<float> cudot (const Darray<float>& lhs, const Darray<float>& rhs)
{
	// context check
	CHECK_EQ(lhs.getDeviceManager().getDeviceID(), rhs.getDeviceManager().getDeviceID());
	
	CHECK_EQ(lhs.ndim(), rhs.ndim());
	CHECK_LT(lhs.ndim(), 3);
	CHECK_LT(rhs.ndim(), 3);

	Darray<float> ret;

	if (lhs.ndim()==1 && rhs.ndim()==1)
	{
		// shape check
		CHECK_EQ(lhs.size(), rhs.size());
		ret = Darray<float>(lhs.getDeviceManager(), {1});
		
		// using cublas sdot
		lhs.deviceSet();
		cublasSdot (DeviceManager::handle,
				    lhs.size(),
				    lhs.data,
				    1,
				    rhs.data,
				    1,
				    ret.data);
	}
	// 2D matrix dot
	else if (lhs.ndim()==2 && rhs.ndim()==2)
	{
		// shape check
		CHECK_EQ(lhs.shape()[1], rhs.shape()[0]);
		ret = Darray<float>(lhs.getDeviceManager(), {lhs.shape()[0], rhs.shape()[1]});
		
		// using cublas sgemm
		lhs.deviceSet();
		const float alpha = 1.;
		const float beta = 0.;
		CUBLAS_SAFE_CALL(
		cublasSgemm (DeviceManager::handle,
					CUBLAS_OP_N,
					CUBLAS_OP_N,
					lhs.shape()[0],
					rhs.shape()[1],
					lhs.shape()[1],
					&alpha,
					lhs.dev_data,
					lhs.shape()[0],
					rhs.dev_data,
					rhs.shape()[0],
					&beta,
					ret.dev_data,
					ret.shape()[0])
		);
	}
	return ret;
}


double cunorm2 (const Darray<double>& ary)
{
	ary.deviceSet();
	double ret;
	CUBLAS_SAFE_CALL(
			cublasDnrm2 (DeviceManager::handle,
						 ary.size(),
						 ary.dev_data,
						 1,
						 &ret)
	);
	return ret;
}


float cunorm2 (const Darray<float>& ary)
{
	ary.deviceSet();
	float ret;
	CUBLAS_SAFE_CALL(
			cublasSnrm2 (DeviceManager::handle,
						 ary.size(),
						 ary.dev_data,
						 1,
						 &ret)
	);
	return ret;
}

} // namespace numcp
#endif
