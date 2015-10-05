#ifndef NUMCP_OPENCV
#define NUMCP_OPENCV

#include <opencv2/opencv.hpp>
#include <numcp.h>
#include <iostream>

namespace numcp {

template<typename T_> std::vector<T_> mat2vec (cv::Mat& src); 
template<typename T_> cv::Mat vec2mat (std::vector<T_>& src); 
template<typename T_> Array<T_> mat2array (cv::Mat& src);
template<typename T_> cv::Mat array2mat (Array<T_>& src); 

template<typename T_> 
std::vector<T_> mat2vec (cv::Mat& src)
{
	std::vector<T_> dst;
	std::vector<cv::Mat> planes;
	cv::split(src, planes);

	for (std::vector<cv::Mat>::reverse_iterator i=planes.rbegin(); i!=planes.rend(); ++i)
	{
		std::vector<T_> tmp;
		(*i) = (*i).t();
		(*i) = (*i).reshape(0,1);
		(*i).copyTo(tmp);
		std::copy(tmp.begin(), tmp.end(), std::back_inserter(dst));
	}
	return dst;
}

template<typename T_> 
cv::Mat vec2mat (std::vector<T_>& src, int c, int r)
{
	cv::Mat dst, tmp;
	std::vector<cv::Mat> planes;
	const int len = std::distance(src.begin(), src.end());
	const int partial_len = (len + c - 1) / c;

	for (typename std::vector<T_>::reverse_iterator i=src.rbegin(); i!=src.rend(); std::advance(i, partial_len))
	{
		std::vector<T_> ch_vec(i, i+partial_len);
		planes.push_back(cv::Mat(ch_vec).t());
	}

	cv::merge(planes, tmp);
	tmp = tmp.reshape(c, r);
	cv::flip(tmp.t(), dst, -1);
	return dst;
}

	
template<typename T_>
Array<T_> mat2array (cv::Mat& src)
{
	int ndim = src.channels()==1 ? 2 : 3;
	std::vector<int> shape(ndim);
	if (ndim == 2)
	{
		shape[0] = src.rows;
		shape[1] = src.cols;
	}
	else
	{
		shape[0] = src.channels();
		shape[1] = src.rows;
		shape[2] = src.cols;
	}

	std::vector<T_> matvec =  mat2vec<T_>(src);
	return Array<T_>(matvec, shape);
}

template<typename T_>
cv::Mat array2mat (Array<T_>& src)
{
	int c = src.ndim()==2 ? 1 : src.shape()[0];
	int r = src.ndim()==2 ? src.shape()[0] : src.shape()[1];
	std::vector<T_> vec = src.vector();
	return vec2mat<T_>(vec, c, r);
}

} // namespace numcp
#endif
