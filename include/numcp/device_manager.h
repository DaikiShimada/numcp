#ifndef NUMCP_DEVICE_MNG
#define NUMCP_DEVICE_MNG

#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "numcp_cblas_helper.h"

namespace numcp {

class DeviceManager
{
public:
	static cublasHandle_t handle;
	static void CreateHandle() {CUBLAS_SAFE_CALL(cublasCreate(&handle));}

	DeviceManager();
	DeviceManager(const int deviceID);
	DeviceManager(const DeviceManager& obj);
	DeviceManager& operator=(const DeviceManager& obj);
	friend std::ostream& operator<<(std::ostream& os, const DeviceManager& rhs);
	~DeviceManager();

	int getDeviceID() const {return deviceID;}
	std::string getDeviceName() const {return deviceProp.name;}
	int getDeviceMajor() const {return deviceProp.major;}
	int getDeviceMinor() const {return deviceProp.minor;}

	void deviceSet() const;
	void deviceReset() const;

private:
	int deviceID;
	cudaError_t e;
	cudaDeviceProp deviceProp;
};
}
#endif
