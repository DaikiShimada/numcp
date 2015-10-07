#ifndef NUMCP_DARRAY
#define NUMCP_DARRAY

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "array.hpp"
#include "numcp_cblas_helper.h"

namespace numcp {

template<typename T_> 
class Darray : Array<T_>
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
Darray<T_>::Darray() : Array<T_>()
{
	dev_size = sizeof(*Array<T_>::data) * Array<T_>::_size;
	cudaMalloc ((void**)&dev_data, dev_size);
}

template<typename T_> 
Darray<T_>::Darray(const std::vector<int>& _shape_) : Array<T_>(_shape_)
{
	dev_size = sizeof(*Array<T_>::data) * Array<T_>::_size;
	cudaMalloc ((void**)&dev_data, dev_size);
}

template<typename T_> 
Darray<T_>::Darray(const std::vector<int>& _shape_, const T_ value) : Array<T_>(_shape_, value)
{
	dev_size = sizeof(*Array<T_>::data) * Array<T_>::_size;
	cudaMalloc ((void**)&dev_data, dev_size);
}

template<typename T_> 
Darray<T_>::Darray(const std::vector<T_>& src, const std::vector<int>& _shape_) : Array<T_>(src, _shape_)
{
	dev_size = sizeof(*Array<T_>::data) * Array<T_>::_size;
	cudaMalloc ((void**)&dev_data, dev_size);
}

template<typename T_> 
Darray<T_>::Darray(const Array<T_>& obj) : Array<T_>(obj)
{
	dev_size = sizeof(*Array<T_>::data) * Array<T_>::_size;
	cudaMalloc ((void**)&dev_data, dev_size);
}

template<typename T_> 
Darray<T_>::Darray(const Darray<T_>& obj) : Array<T_>(obj)
{
	this->dev_size = obj.dev_size;
	cudaMalloc ((void**)&this->dev_data, this->dev_size);
}

template<typename T_> 
Darray<T_>& Darray<T_>::operator=(const Darray<T_>& obj)
{
	Array<T_>::operator=(obj);
	this->dev_size = obj.dev_size;
	cudaMalloc ((void**)&this->dev_data, this->dev_size);
	return (*this);
}

template<typename T_> 
Darray<T_>::~Darray()
{
	cudaFree(dev_data);
}

/* public functions */
template<typename T_> 
void Darray<T_>::to_device()
{
	CUDA_SAFE_CALL(cudaMemcpy(dev_data, Array<T_>::data, dev_size, cudaMemcpyHostToDevice)); 
}


template<typename T_> 
void Darray<T_>::to_host()
{
	CUDA_SAFE_CALL(cudaMemcpy(Array<T_>::data, dev_data, dev_size, cudaMemcpyDeviceToHost)); 
}
} // namespace numcp

#endif
