#include <iostream>
#include <vector>
#include <../include/numcp.h>


int main(int argc, char const* argv[])
{
	std::vector<int> shape_a{2, 2, 3};
	std::vector<int> shape_b{2, 2, 3};
	numcp::Array<double> a(shape_a, 0);
	numcp::Array<double> b(shape_b, 1);
	a = a + 10;
	std::cout << -a << std::endl;
	a -= b;
	std::cout << a << std::endl;
	b -= 9;
	b[{0,1,2}] = 999.;
	std::cout << b << std::endl;

	std::cout << b[{0,1,2}] << std::endl;
	
	std::vector<int> shape_c{2, 2, 3, 3};
	numcp::Array<double> c(shape_c, 0);
	std::cout << c << std::endl;
	return 0;
}
