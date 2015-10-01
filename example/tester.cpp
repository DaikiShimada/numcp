#include <vector>
#include <../include/numcp.h>


int main(int argc, char const* argv[])
{
	std::vector<int> shape_a{2, 2};
	std::vector<int> shape_b{3, 2};
	numcp::Array<double> a(shape_a);
	numcp::Array<double> b(shape_b);
	a = a + 10;
	a = 10. + a;
	
	return 0;
}
