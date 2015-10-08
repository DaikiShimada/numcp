#ifndef NUMCP_CUDA_HELPER
#define NUMCP_CUDA_HELPER


#define CUDA_SAFE_CALL(func) \
do { \
	cudaError_t err = (func); \
	if (err != cudaSuccess) { \
		fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
		exit(err); \
	} \
} while(0)

#define CUBLAS_SAFE_CALL(func) \
do { \
	cublasStatus_t err = (func); \
	if (err != CUBLAS_STATUS_SUCCESS) {\
		switch (err) { \
			case CUBLAS_STATUS_NOT_INITIALIZED: \
				fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", "The cuBLAS library was not initialized.", err, __FILE__, __LINE__); \
				exit (err); \
			case CUBLAS_STATUS_ALLOC_FAILED: \
				fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", "Resource allocation failed inside the cuBLAS library.", err, __FILE__, __LINE__); \
				exit (err); \
			case CUBLAS_STATUS_INVALID_VALUE: \
				fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", "An unsupported value or parameter was passed to the function (a negative vector size, for example)", err, __FILE__, __LINE__); \
				exit (err); \
			case CUBLAS_STATUS_ARCH_MISMATCH: \
				fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", "The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.", err, __FILE__, __LINE__); \
				exit (err); \
			case CUBLAS_STATUS_MAPPING_ERROR: \
				fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.", err, __FILE__, __LINE__); \
				exit (err); \
			case CUBLAS_STATUS_EXECUTION_FAILED: \
				fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.", err, __FILE__, __LINE__); \
				exit (err); \
			case CUBLAS_STATUS_INTERNAL_ERROR: \
				fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", "An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.", err, __FILE__, __LINE__); \
				exit (err); \
			case CUBLAS_STATUS_NOT_SUPPORTED: \
				fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", "The functionnality requested is not supported.", err, __FILE__, __LINE__); \
				exit (err); \
			case CUBLAS_STATUS_LICENSE_ERROR: \
				fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", "The functionnality requested requires some license and an error was detected when trying to check the current licensing.", err, __FILE__, __LINE__); \
				exit (err); \
			default: \
				fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", "Unknown error.", err, __FILE__, __LINE__); \
				exit (err); \
		} \
	} \
} while(0)


#endif
