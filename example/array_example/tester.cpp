#include <iostream>
#include <vector>
#include "numcp.h"


int main(int argc, char const* argv[])
{
	std::vector<int> shape_a{2, 3};
	numcp::Array<double> a(shape_a, 0);
	for (int i=0; i<a.size(); ++i) a.data[i] = i/2.;

	std::vector<int> shape_b{3, 1};
	numcp::Array<double> b(shape_b, 0);
	for (int i=0; i<b.size(); ++i) b.data[i] = i+1./4;

	numcp::Array<double> c = numcp::dot(a,b);

	std::cout << a << std::endl;
	std::cout << numcp::mean(a) << std::endl;
	std::cout << numcp::min(a) << std::endl;
	std::cout << numcp::max(a) << std::endl;

	/*	
	std::cout << a << std::endl;
	std::cout << b << std::endl;
	std::cout << numcp::dot(a, b) << std::endl;
	*/
	return 0;
}
