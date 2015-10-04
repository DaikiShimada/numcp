#include <iostream>
#include <vector>
#include <../include/numcp.h>


int main(int argc, char const* argv[])
{
	std::vector<int> shape_a{6, 4};
	numcp::Array<double> a(shape_a, 0);
	for (int i=0; i<a.size(); ++i) a.data[i] = i;
	std::cout << a << std::endl;
	std::cout << a.swapaxes(0,1) << std::endl;
	std::cout << a.T() << std::endl;
	return 0;
}
