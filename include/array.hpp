#ifndef NUMCP_ARRAY
#define NUMCP_ARRAY

#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <glog/logging.h>

namespace numcp {


template<typename T_>
class Array
{
public:
	T_* data;
	int size() const {return _size;}
	int ndim() const {return _ndim;}
	std::vector<int> shape() const {return _shape;}
	std::vector<T_> vector() const;
	const int adr(const std::vector<int> idx) const;
	Array<T_> swapaxes(const int axis_1, const int axis_2) const;
	Array<T_> T() const;

	Array();
	Array(const std::vector<int>& _shape_);
	Array(const std::vector<int>& _shape_, const T_ value);
	Array(const std::vector<T_>& src, const std::vector<int>& _shape_);
	Array(const Array<T_>& obj);
	Array<T_>& operator=(const Array<T_>& obj);
	~Array();

	/* operators */
	// +
	const Array<T_> operator+(const Array<T_>& rhs) const;
	const Array<T_> operator+(const T_& rhs) const;
	template<typename U_> friend Array<U_> operator+(U_ lhs, Array<U_> rhs);
	Array<T_>& operator+=(const Array<T_>& rhs);
	Array<T_>& operator+=(const T_& rhs);
	Array<T_>& operator++();
	const Array<T_> operator++(int notused);
	// -
	const Array<T_> operator-(const Array<T_>& rhs) const;
	const Array<T_> operator-(const T_& rhs) const;
	const Array<T_> operator-() const;
	template<typename U_> friend Array<U_> operator-(U_& lhs, Array<U_>& rhs);
	Array<T_>& operator-=(const Array<T_>& rhs);
	Array<T_>& operator-=(const T_& rhs);
	Array<T_>& operator--();
	const Array<T_> operator--(int notused);
	// *
	const Array<T_> operator*(const Array<T_>& rhs) const;
	const Array<T_> operator*(const T_& rhs) const;
	template<typename U_> friend Array<U_> operator*(U_& lhs, Array<U_>& rhs);
	Array<T_>& operator*=(const Array<T_>& rhs);
	Array<T_>& operator*=(const T_& rhs);
	// /
	const Array<T_> operator/(const Array<T_>& rhs) const;
	const Array<T_> operator/(const T_& rhs) const;
	template<typename U_>friend Array<U_> operator/(U_& lhs, Array<U_>& rhs);
	Array<T_>& operator/=(const Array<T_>& rhs);
	Array<T_>& operator/=(const T_& rhs);
	// boolean
	bool operator==(const Array<T_>& rhs) const;	
	bool operator!=(const Array<T_>& rhs) const;	
	// []
	const T_& operator[](const std::vector<int> idx) const;
	T_& operator[](const std::vector<int> idx);
	// <<
	template<typename U_> friend std::ostream& operator<<(std::ostream& os, const Array<U_>& rhs);

private:
	int _size;	//! Arrayの要素数, _shapeの総積に等しい
	int _ndim;	//! Arrayの次元数
	std::vector<int> _shape;	//! Arrayの形状, {..., チャネル, 行数, 列数}
	std::vector<int> _memshape;	//! メモリ上でのArrayの形状, {..., チャネル, 列数, 行数}

	void initMemshape();
	void checkSameShape(const Array<T_>& rhs) const;

};

template<typename T_> 
std::ostream& recursive_array_disp(const Array<T_>& ary, std::vector<int>& ary_shape, std::vector<int>& idx, int dim);



template<typename T_> 
Array<T_>::Array()
{
	_size = 1;
	_shape = std::vector<int>(1);
	_ndim = _shape.size();
	data = new T_[_size];

	initMemshape();
}


template<typename T_> 
Array<T_>::Array(const std::vector<int>& _shape_)
{
	this->_shape = _shape_;
	this->_ndim = _shape_.size();
	this->_size = std::accumulate(_shape_.begin(), _shape_.end(), 1, std::multiplies<int>());
	this->data = new T_[_size];

	initMemshape();
}



template<typename T_> 
Array<T_>::Array(const std::vector<int>& _shape_, const T_ value)
{
	this->_shape = _shape_;
	this->_ndim = _shape_.size();
	this->_size = std::accumulate(_shape_.begin(), _shape_.end(), 1, std::multiplies<int>());
	this->data = new T_[_size];
	for (int i=0; i<this->_size; ++i) this->data[i] = value;

	initMemshape();
}

template<typename T_> 
Array<T_>::Array(const std::vector<T_>& src, const std::vector<int>& _shape_)
{
	this->_shape = _shape_;
	this->_ndim = _shape_.size();
	this->_size = std::accumulate(_shape_.begin(), _shape_.end(), 1, std::multiplies<int>());
	CHECK_EQ(this->_size, src.size());
	this->data = new T_[_size];
	for (int i=0; i<this->_size; ++i) this->data[i] = src.at(i);
	
	initMemshape();
}


template<typename T_> 
Array<T_>::Array(const Array<T_>& obj)
{
	this->_size = obj._size;
	this->_ndim = obj._ndim;
	this->_shape = obj._shape;
	this->data = new T_[this->_size];
	for (int i=0; i<this->_size; ++i) this->data[i] = obj.data[i];
}



template<typename T_> 
Array<T_>& Array<T_>::operator=(const Array<T_>& obj)
{
	this->_size = obj._size;
	this->_ndim = obj._ndim;
	this->_shape = obj._shape;
	this->_memshape = obj._memshape;
	this->data = new T_[this->_size];
	for (int i=0; i<this->_size; ++i) this->data[i] = obj.data[i];

	return(*this);
}


template<typename T_> 
Array<T_>::~Array()
{
	delete[] data;
	std::vector<int>().swap(_shape);
	std::vector<int>().swap(_memshape);
}


/* public functions */
template<typename T_> 
std::vector<T_> Array<T_>::vector() const
{
	return std::vector<T_>(data, data+_size);
}


template<typename T_> 
const int Array<T_>::adr(const std::vector<int> idx) const
{
	// check idx
	CHECK_EQ(idx.size(), _ndim);
	for (int i=0; i<_ndim; ++i)
	{
		CHECK_LT(idx[i], _shape[i]);
	}

	// return adr indx of data
	if (idx.size() == 1) return idx[0];

	std::vector<int> memidx (idx);
	if (memidx.size() >= 2)
	{
		int tmp = memidx[memidx.size()-1];
		memidx[memidx.size()-1] = memidx[memidx.size()-2];
		memidx[memidx.size()-2] = tmp;
	}

	int ret = 0;
	int past_dim = 1;
	for (int i=0; i<memidx.size(); ++i)
	{
		ret += past_dim * memidx[memidx.size() -1 - i]; 
		past_dim *= _memshape[memidx.size() -1 - i];
	}

	return ret;
}

template<typename T_> 
Array<T_> Array<T_>::swapaxes(const int axis_1, const int axis_2) const
{
	CHECK_LE(2, _ndim);
	std::vector<int> swshape (_shape);
	int tmp = swshape[axis_1];
	swshape[axis_1] = swshape[axis_2];
	swshape[axis_2] = tmp;

	Array<T_> swary (swshape);
	std::vector<int> idx(_ndim, 0);

	std::function<void (int)> f;
	f = [&](int dim)
	{
		for (idx[dim]=0; idx[dim]<_shape[dim]; ++idx[dim])
		{
			if (dim == _ndim-1)
			{
				std::vector<int> sw_idx(idx);
				sw_idx[axis_1] = idx[axis_2];
				sw_idx[axis_2] = idx[axis_1];
				swary[sw_idx] = data[adr(idx)];
			}
			else
			{
				f(dim+1);
			}
		}
		idx[dim] = 0;
	};
	
	f(0);

	return swary;
}


template<typename T_> 
Array<T_> Array<T_>::T() const
{
	CHECK_LE(2, _ndim);
	return this->swapaxes(0, _ndim-1);	
}

/* public functions */
/*
const Array<T_> T() const
{
	Array<T_> ret();
	return ret;
}
*/

/* operators */
// +
template<typename T_> 
const Array<T_> Array<T_>::operator+(const Array<T_>& rhs) const
{
	this->checkSameShape(rhs);	
	Array<T_> ret(this->_shape);
	for (int i=0; i<_size; ++i) ret.data[i] = data[i] + rhs.data[i];
	return ret;
}


template<typename T_> 
const Array<T_> Array<T_>::operator+(const T_& rhs) const
{
	Array<T_> ret(_shape);
	for (int i=0; i<_size; ++i) ret.data[i] = data[i] + rhs;
	return ret;
}



template<typename T_> 
Array<T_> operator+(T_ lhs, Array<T_> rhs)
{
	Array<T_> ret(rhs.shape());
	for (int i=0; i<rhs.size(); ++i) ret.data[i] = lhs + rhs.data[i];
	return ret;
}



template<typename T_> 
Array<T_>& Array<T_>::operator+=(const Array<T_>& rhs)
{
	checkSameShape(rhs);
	for (int i=0; i<_size; ++i) this->data[i] += rhs.data[i];
	return (*this);
}


template<typename T_> 
Array<T_>& Array<T_>::operator+=(const T_& rhs)
{
	for (int i=0; i<_size; ++i) data[i] += rhs;
	return (*this);
}


template<typename T_> 
Array<T_>& Array<T_>::operator++()
{
	for (int i=0; i<_size; ++i) data[i]++;
	return (*this);
}



template<typename T_> 
const Array<T_> Array<T_>::operator++(int notused)
{
	const Array<T_> tmp = *this;
	++(*this);
	return tmp;
}

// -
template<typename T_> 
const Array<T_> Array<T_>::operator-(const Array<T_>& rhs) const
{
	checkSameShape(rhs);
	Array<T_> ret(_shape);
	for (int i=0; i<_size; ++i) ret.data[i] = data[i] - rhs.data[i];
	return ret;
}


template<typename T_> 
const Array<T_> Array<T_>::operator-(const T_& rhs) const
{
	Array<T_> ret(_shape);
	for (int i=0; i<_size; ++i) ret.data[i] = data[i] - rhs;
	return ret;
}



template<typename T_> 
const Array<T_> Array<T_>::operator-() const
{
	Array<T_> ret(_shape);
	for (int i=0; i<_size; ++i) ret.data[i] = -data[i];
	return ret;
}



template<typename T_>
Array<T_> operator-(T_ lhs, Array<T_> rhs)
{
	Array<T_> ret(rhs.shape());
	for (int i=0; i<ret._size; ++i) ret.data[i] = lhs - rhs.data[i];
	return ret;
}



template<typename T_> 
Array<T_>& Array<T_>::operator-=(const Array<T_>& rhs)
{
	checkSameShape(rhs);
	for (int i=0; i<_size; ++i) data[i] -= rhs.data[i];
	return (*this);
}


template<typename T_> 
Array<T_>& Array<T_>::operator-=(const T_& rhs)
{
	for (int i=0; i<_size; ++i) data[i] -= rhs;
	return (*this);
}


template<typename T_> 
Array<T_>& Array<T_>::operator--()
{
	for (int i=0; i<_size; ++i) data[i]--;
	return (*this);
}


template<typename T_> 
const Array<T_> Array<T_>::operator--(int notused)
{
	const Array<T_> tmp = *this;
	--(*this);
	return tmp;
}


// *
template<typename T_>
const Array<T_> Array<T_>::operator*(const Array<T_>& rhs) const
{
	checkSameShape(rhs);
	Array<T_> ret(_shape);
	for (int i=0; i<ret._size; ++i) ret.data[i] = data[i] * rhs.data[i];
	return ret;
}


template<typename T_>
const Array<T_> Array<T_>::operator*(const T_& rhs) const
{
	Array<T_> ret(_shape);
	for (int i=0; i<ret._size; ++i) ret.data[i] = data[i] * rhs;
	return ret;
}


template<typename T_>
Array<T_> operator*(T_& lhs, Array<T_>& rhs)
{
	Array<T_> ret(rhs.shape());
	for (int i=0; i<ret._size; ++i) ret.data[i] = lhs * rhs.data[i];
	return ret;
}


template<typename T_>
Array<T_>& Array<T_>::operator*=(const Array<T_>& rhs)
{
	checkSameShape(rhs);
	for (int i=0; i<_size; ++i) data[i] *= rhs.data[i];
	return (*this);
}


template<typename T_>
Array<T_>& Array<T_>::operator*=(const T_& rhs)
{
	for (int i=0; i<_size; ++i) data[i] *= rhs;
	return (*this);
}





// /
template<typename T_>
const Array<T_> Array<T_>::operator/(const Array<T_>& rhs) const
{
	checkSameShape(rhs);
	Array<T_> ret(_shape);
	for (int i=0; i<ret._size; ++i) ret.data[i] = data[i] / rhs.data[i];
	return ret;
}



template<typename T_>
const Array<T_> Array<T_>::operator/(const T_& rhs) const
{
	Array<T_> ret(_shape);
	for (int i=0; i<ret._size; ++i) ret.data[i] = data[i] / rhs;
	return ret;
}


template<typename T_>
Array<T_> operator/(T_& lhs, Array<T_>& rhs)
{
	Array<T_> ret(rhs.shape());
	for (int i=0; i<ret.size(); ++i) ret.data[i] = lhs / rhs.data[i];
	return ret;
}


template<typename T_>
Array<T_>& Array<T_>::operator/=(const Array<T_>& rhs)
{
	checkSameShape(rhs);
	for (int i=0; i<_size; ++i) data[i] /= rhs.data[i];
	return (*this);
}


template<typename T_>
Array<T_>& Array<T_>::operator/=(const T_& rhs)
{
	for (int i=0; i<_size; ++i) data[i] /= rhs;
	return (*this);
}

// boolean operations
template<typename T_>
bool Array<T_>::operator==(const Array<T_>& rhs) const
{
	if (_ndim!=rhs._ndim || _size!=rhs._size) return false;
	for (int i=0; i<_ndim; ++i)
	{
		if (_shape[i] != rhs._shape[i]) return false;
	}

	for (int i=0; i<_size; ++i)
	{
		if (data[i] != rhs.data[i]) return false;
	}
	return true;
}

template<typename T_>
bool Array<T_>::operator!=(const Array<T_>& rhs) const
{
	return !((*this)==rhs);
}

// access operations
template<typename T_>
const T_& Array<T_>::operator[](const std::vector<int> idx) const
{
	return data[adr(idx)];
}


template<typename T_>
T_& Array<T_>::operator[](const std::vector<int> idx)
{
	return data[adr(idx)];
}



// <<
template<typename T_>
std::ostream& operator<<(std::ostream& os, const Array<T_>& rhs)
{
	std::vector<int> idx (rhs.ndim(), 0);
	std::vector<int> rhs_shape = rhs.shape();
	int d = 0;

	return recursive_array_disp(rhs, rhs_shape, idx, d, os);
}


template<typename T_> 
std::ostream& recursive_array_disp(const Array<T_>& ary, std::vector<int>& ary_shape, std::vector<int>& idx, int dim, std::ostream& os)
{
	os << "[";
	if (dim == ary_shape.size()-1)
	{
		for (idx[dim]=0; idx[dim]<ary_shape[dim]; ++idx[dim])
		{
			// start for debug
//			for (std::vector<int>::iterator i=idx.begin(); i!=idx.end(); ++i)
//				os << *i << "/";
//			os << "  ";
			// end for debug

			os << ary[idx];
			if (idx[dim] != ary_shape[dim]-1) os << ", ";
		}
		idx[dim] = 0;
	}
	else
	{
		for (idx[dim]=0; idx[dim]<ary_shape[dim]; ++idx[dim])
		{
			recursive_array_disp(ary, ary_shape, idx, dim+1, os);
			if (idx[dim] != ary_shape[dim]-1) os << ", " << std::endl;
			if (dim == 0 && idx[dim]!=ary_shape[dim]-1) os << std::endl;
		}
		idx[dim] = 0;
	}
	os << "]";
	return os;
}

/* private finction */
template<typename T_> 
void Array<T_>::checkSameShape(const Array<T_>& rhs) const
{
	CHECK_EQ(_ndim, rhs._ndim);
	for (int i=0; i<_ndim; ++i)
		CHECK_EQ(_shape[i], rhs._shape[i]);
}

template<typename T_>
void Array<T_>::initMemshape()
{
	_memshape = std::vector<int>(_shape);
	if (_memshape.size() >= 2)
	{
		int tmp = _memshape[_memshape.size()-1];
		_memshape[_memshape.size()-1] = _memshape[_memshape.size()-2];
		_memshape[_memshape.size()-2] = tmp;
	}
}

}

#endif
