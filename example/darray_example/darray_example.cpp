#include <bench.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <glog/logging.h>
#include <numcp_cuda.hpp>

int main(int argc, char const* argv[])
{
	int deviceID = 0;
	numcp::DeviceManager dev_mng(deviceID);
	numcp::DeviceManager::CreateHandle();
	std::cout << dev_mng << std::endl;
	
	numcp::Darray<float> as(dev_mng, {300,300});
	numcp::Darray<double> ad(dev_mng, {300,300});
	numcp::Darray<float> bs(dev_mng, {300,300});
	numcp::Darray<double> bd(dev_mng, {300,300});

	benchmark("CPU(double) ver.") { numcp::dot(ad, bd); }
	benchmark("GPU(float) ver.") 
	{ 
		as.to_device();
		bs.to_device();
		numcp::cudot(as, bs);
	}
	benchmark("GPU(double) ver.") { numcp::cudot(ad, bd); }

	dev_mng.deviceReset();

	return 0;
}
