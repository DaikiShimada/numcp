#ifndef NUMCP_DARRAY
#define NUMCP_DARRAY

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "array.hpp"
#include "numcp_cblas_helper.h"

namespace numcp {

template<typename T_> 
class Darray : public Array<T_>
{
public:
	T_* dev_data;

	Darray();
	Darray(const DeviceManager& mng, const std::vector<int>& _shape_);
	Darray(const DeviceManager& mng, const std::vector<int>& _shape_, const T_ value);
	Darray(const DeviceManager& mng, const std::vector<T_>& src, const std::vector<int>& _shape_);
	Darray(const DeviceManager& mng, const Array<T_>& obj);
	Darray(const Darray<T_>& obj);
	Darray<T_>& operator=(const Darray<T_>& obj);
	~Darray();

	std::size_t getDev_size() const {return dev_size; }
	const DeviceManager& getDeviceManager() const { return dev_mng; }

	void deviceSet() const {dev_mng.deviceSet();}
	void to_device();
	void to_host();

private:
	std::size_t dev_size;
	DeviceManager dev_mng;
};


template<typename T_> 
Darray<T_>::Darray() : Array<T_>()
{
	dev_mng = DeviceManager(0);
	dev_size = sizeof(*Array<T_>::data) * Array<T_>::_size;
	dev_mng.deviceSet();
	cudaMalloc ((void**)&dev_data, dev_size);
}

template<typename T_> 
Darray<T_>::Darray(const DeviceManager& mng, const std::vector<int>& _shape_) : Array<T_>(_shape_)
{
	dev_mng = mng;
	dev_size = sizeof(*Array<T_>::data) * Array<T_>::_size;
	dev_mng.deviceSet();
	cudaMalloc ((void**)&dev_data, dev_size);
}

template<typename T_> 
Darray<T_>::Darray(const DeviceManager& mng, const std::vector<int>& _shape_, const T_ value) : Array<T_>(_shape_, value)
{
	dev_mng = mng;
	dev_size = sizeof(*Array<T_>::data) * Array<T_>::_size;
	dev_mng.deviceSet();
	cudaMalloc ((void**)&dev_data, dev_size);
}

template<typename T_> 
Darray<T_>::Darray(const DeviceManager& mng, const std::vector<T_>& src, const std::vector<int>& _shape_) : Array<T_>(src, _shape_)
{
	dev_mng = mng;
	dev_size = sizeof(*Array<T_>::data) * Array<T_>::_size;
	dev_mng.deviceSet();
	cudaMalloc ((void**)&dev_data, dev_size);
}

template<typename T_> 
Darray<T_>::Darray(const DeviceManager& mng, const Array<T_>& obj) : Array<T_>(obj)
{
	dev_mng = mng;
	dev_size = sizeof(*Array<T_>::data) * Array<T_>::_size;
	dev_mng.deviceSet();
	cudaMalloc ((void**)&dev_data, dev_size);
}

template<typename T_> 
Darray<T_>::Darray(const Darray<T_>& obj) : Array<T_>(obj)
{
	this->dev_mng = obj.getDeviceManager();
	this->dev_size = obj.dev_size;
	dev_mng.deviceSet();
	cudaMalloc ((void**)&this->dev_data, this->dev_size);
	CUDA_SAFE_CALL(cudaMemcpy(this->dev_data, obj.dev_data, dev_size, cudaMemcpyDeviceToDevice)); 
}

template<typename T_> 
Darray<T_>& Darray<T_>::operator=(const Darray<T_>& obj)
{
	Array<T_>::operator=(obj);
	this->dev_mng = obj.getDeviceManager();
	this->dev_size = obj.getDev_size();
	dev_mng.deviceSet();
	cudaMalloc ((void**)&this->dev_data, this->dev_size);
	CUDA_SAFE_CALL(cudaMemcpy(this->dev_data, obj.dev_data, dev_size, cudaMemcpyDeviceToDevice)); 
	return (*this);
}

template<typename T_> 
Darray<T_>::~Darray()
{
	dev_mng.deviceSet();
	cudaFree(dev_data);
}

/* public functions */
template<typename T_> 
void Darray<T_>::to_device()
{
	dev_mng.deviceSet();
	CUDA_SAFE_CALL(cudaMemcpy(dev_data, Array<T_>::data, dev_size, cudaMemcpyHostToDevice)); 
}


template<typename T_> 
void Darray<T_>::to_host()
{
	dev_mng.deviceSet();
	CUDA_SAFE_CALL(cudaMemcpy(Array<T_>::data, dev_data, dev_size, cudaMemcpyDeviceToHost)); 
}
} // namespace numcp

#endif
