#ifndef NUMCP_DARRAY
#define NUMCP_DARRAY

#include "array.hpp"

namespace numcp {

template<typename T_> 
class Darray : public Array
{
public:
	T_* dev_data;

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
};

} // namespace numcp
#endif
