#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H

#include <iostream>
#include <opencv2/opencv.hpp>

/*
 *  * * * * * * *   pyramid functions * * * * * * * * * * * * *
*/
template<typename T>
static void pyrDownMean(const cv::Mat& in, cv::Mat& out)
{
    out.create(cv::Size( in.size().width /2 , in.size().height /2 ), in.type() );
    for(int v=0; v< out.rows; v++)
    {
        for(int u =0; u< out.cols; u++)
        {
            int u0 = u*2;
            int u1 = u0 + 1;
            int v0 = v*2;
            int v1 = v0+1;
            out.ptr<T>(v)[u] = (T)( (in.ptr<T>(v0)[u0] + in.ptr<T>(v0)[u1] + in.ptr<T>(v1)[u0] + in.ptr<T>(v1)[u1]) / 4.0f);
        }
    }
}

// pyrDownMedian is used to downsample depth img.
// we want to retain the depth origin value other than bring in a new depth value by pyrDownMean
template<typename T>
static void pyrDownMedian(const cv::Mat& in, cv::Mat& out)
{
    cv::Mat in_median;
    cv::medianBlur(in,in_median,3);
    out.create(cv::Size( in.size().width /2 , in.size().height /2 ), in.type() );
    for(int v=0; v< out.rows; v++)
    {
        for(int u =0; u< out.cols; u++)
        {
            out.ptr<T>(v)[u] = in_median.ptr<T>(v*2)[u*2];
        }
    }
}

template<typename T>
static void calculateDerivativeX(const cv::Mat& in, cv::Mat& dx_result)
{
    dx_result.create( in.size(), in.type() );
    for(int v=0; v< in.rows; v++)
    {
        for(int u =0; u< in.cols; u++)
        {
            int prev = std::max(u-1 , 0);
            int next = std::min(u+1 , in.cols - 1);
            // dx = (img(x+1) - img(x) + img(x) - img(x-1) ) / 2
            dx_result.ptr<T>(v)[u] = ( in.ptr<T>(v)[next] -  in.ptr<T>(v)[prev] ) * 0.5f;
        }
    }
}

template<typename T>
static void calculateDerivativeY(const cv::Mat& in, cv::Mat& dy_result)
{
    dy_result.create( in.size(), in.type() );
    for(int v=0; v< in.rows; v++)
    {
        for(int u =0; u< in.cols; u++)
        {
            int prev = std::max(v-1 , 0);
            int next = std::min(v+1 , in.rows - 1);
            dy_result.ptr<T>(v)[u] = (in.ptr<T>(next)[u] -  in.ptr<T>(prev)[u] ) * 0.5f ;
        }
    }
}

/*
 * bilinearWithdepth :
 *        if depth(x,y) -  depth(xi,yi) >  threshold
 *             says the pixel(xi,yi) may be not the same kind of pixel as pixel(x,y),
 *        then
 *             the pixel(xi,yi) should not be counted in
*/
template<typename T>
T bilinearWithdepth(const cv::Mat& img, const cv::Mat& depth, const float& x,const float& y, const float& z)
{
    int x0 = (int)std::floor(x);
    int y0 = (int )std::floor(y);
    int x1 = x0 + 1;
    int y1 =  y0+ 1;

    if(x1 >= img.cols || y1 >= img.rows)
    {
        return std::numeric_limits<T>::quiet_NaN();
    }

    float x0_weight = x1 - x;
    float y0_weight = y1 - y;
    float x1_weight = 1.0f - x0_weight;
    float y1_weight = 1.0f - y0_weight;
    float z_eps = z - 0.05f ;   // 0.05 is 50cm threshold

    float val(0.0f), sum(0.0f);

    if( std::isfinite(depth.at<float>(y0,x0)) && depth.at<float>(y0,x0) > z_eps )
    {
        val += x0_weight * y0_weight * img.at< T >(y0,x0);
        sum += x0_weight * y0_weight;
    }

    if( std::isfinite(depth.at<float>(y0,x1)) && depth.at<float>(y0,x1) > z_eps )
    {
        val += x1_weight * y0_weight * img.at< T >(y0,x1);
        sum += x1_weight * y0_weight;
    }

    if( std::isfinite(depth.at<float>(y1,x0)) && depth.at<float>(y1,x0) > z_eps )
    {
        val += x0_weight * y1_weight * img.at< T >(y1 , x0);
        sum += x0_weight * y1_weight;
    }

    if( std::isfinite(depth.at<float>(y1,x1)) && depth.at<float>(y1,x1) > z_eps )
    {
        val += x1_weight * y1_weight * img.at< T >(y1,x1);
        sum += x1_weight * y1_weight;
    }

    if( sum > 0.0f )
    {
        return (T) val/sum ;
    }else
    {
        return std::numeric_limits<T>::quiet_NaN();
    }
}

template<typename T>
T bilinear(const cv::Mat& img, const float& x, const float& y)
{
    int x0 = (int)std::floor(x);
    int y0 = (int )std::floor(y);
    int x1 = x0 + 1;
    int y1 =  y0 + 1;

    float x0_weight = x1 - x;
    float y0_weight = y1 - y;
    float x1_weight = 1.0f - x0_weight;
    float y1_weight = 1.0f - y0_weight;

    if(x1 >= img.cols || y1 >= img.rows)
    {
        return std::numeric_limits<T>::quiet_NaN();
    }

    float interpolated =
            img.at<T>(y0 , x0 ) * x0_weight + img.at<T>(y0 , x1)* x1_weight +
            img.at<T>(y1 , x0 ) * x0_weight + img.at<T>(y1 , x1)* x1_weight +
            img.at<T>(y0 , x0 ) * y0_weight + img.at<T>(y1 , x0)* y1_weight +
            img.at<T>(y0 , x1 ) * y0_weight + img.at<T>(y1 , x1)* y1_weight ;

  return (T)(interpolated * 0.25f);
}
#endif // IMAGEPROCESS_H
