#ifndef NUMCP_ARRAY
#define NUMCP_ARRAY

#include <vector>
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
	const int convertToAdr(const std::vector<int> idx) const;
	const Array<T_> T() const;
	const Array<T_> at(std::vector<int> idx) const;

	Array();
	Array(const std::vector<int>& _shape_);
	Array(const std::vector<int>& _shape_, const T_ value);
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
const int Array<T_>::convertToAdr(const std::vector<int> idx) const
{
	// check idx
	CHECK_EQ(idx.size(), _ndim);
	for (int i=0; i<_ndim; ++i)
	{
		CHECK_LT(idx[i], _shape[i]);
	}

	// return adr indx of data
	
}


template<typename T_> 
const Array<T_> Array<T_>::T() const
{
	
}



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



// <<
template<typename T_>
std::ostream& operator<<(std::ostream& os, const Array<T_>& rhs)
{
	for (int i=0; i<rhs.size(); ++i)
		os << rhs.data[i] << ", ";
	return os;
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
	if (_memshape.size() > 2)
	{
		int tmp = _memshape[_memshape.size()-1];
		_memshape[_memshape.size()-1] = _memshape[_memshape.size()-2];
		_memshape[_memshape.size()-2] = tmp;
	}
}

}

#endif
