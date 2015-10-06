#ifndef NUMCP_DARRAY
#define NUMCP_DARRAY

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "numcp_cblas_helper.h"
#include "array.hpp"

namespace numcp {

template<typename T_> 
class Darray : public Array
{
public:
	T_* dev_data;
	std::size_t dev_size;

	Darray();
	Darray(const std::vector<int>& _shape_);
	Darray(const std::vector<int>& _shape_, const T_ value);
	Darray(const std::vector<T_>& src, const std::vector<int>& _shape_);
	Darray(const Array<T_>& obj);
	Darray(const Darray<T_>& obj);
	Darray<T_>& operator=(const Darray<T_>& obj);
	~Darray();

	void to_device();
	void to_host();

private:
	DeviceManager* dev_mng;
};


template<typename T_> 
Darray()
{
	std::size_t dev_size = sizeof(*data) * _size;
	cudaMalloc ((void**)&dev_data, dev_size);
}

/* public functions */
template<typename T_> 
void Array<T_>::to_device()
{
	CUDA_SAFE_CALL(cudaMemcpy(dev_data, data, dev_size, cudaMemcpyHostToDevice); 
}


} // namespace numcp
#endif
