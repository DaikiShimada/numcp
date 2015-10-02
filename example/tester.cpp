#include <iostream>
#include <vector>
#include <../include/numcp.h>


int main(int argc, char const* argv[])
{
	std::vector<int> shape_a{2, 2};
	std::vector<int> shape_b{2, 2};
	numcp::Array<double> a(shape_a, 0);
	numcp::Array<double> b(shape_b, 1);
	a = a + 10;
	std::cout << -a << std::endl;
	a -= b;
	std::cout << a << std::endl;
	b -= 9;
	std::cout << b << std::endl;
	
	return 0;
}
