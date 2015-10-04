#include <iostream>
#include <vector>
#include <../include/numcp.h>


int main(int argc, char const* argv[])
{
	std::vector<int> shape_a{200, 3000};
	numcp::Array<double> a(shape_a, 0);
	for (int i=0; i<a.size(); ++i) a.data[i] = i/2;

	std::vector<int> shape_b{3000, 100};
	numcp::Array<double> b(shape_b, 0);
	for (int i=0; i<b.size(); ++i) b.data[i] = i+1./4;

	numcp::Array<double> c = numcp::dot(a,b);
	std::cout << c[{3,10}] << std::endl;

	/*	
	std::cout << a << std::endl;
	std::cout << b << std::endl;
	std::cout << numcp::dot(a, b) << std::endl;
	*/
	return 0;
}
