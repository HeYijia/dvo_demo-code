#ifndef FRAMEDATA_H
#define FRAMEDATA_H

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

#include <Eigen/Geometry>

void saveMatToCsv(cv::Mat &data, std::string filename);
void saveEigenToCsv(Eigen::MatrixXf matrix, std::string filename);

typedef Eigen::Transform<float,3,Eigen::Affine> AffineTransform;

namespace dvo_framedata {

struct IntrinsicMatrix
{
    float depth_factor;
    float ox;
    float oy;
    float fx;
    float fy;
};

struct FrameRGBD
{
    std::string frameID;
    typedef Eigen::Matrix<float,4,Eigen::Dynamic,Eigen::ColMajor> xyzPointCloud; // stock the xyz 3D-coordinate
    xyzPointCloud _xyzclouds;
    cv::Mat _rgb;    // used to debug
    cv::Mat _grey;  // used to debug
    cv::Mat _intensity;
    cv::Mat _intensity_dx;
    cv::Mat _intensity_dy;

    cv::Mat _dpt;
    cv::Mat _dpt_16u; // used to debug
    cv::Mat _dpt_dx;
    cv::Mat _dpt_dy;

    IntrinsicMatrix _intrinsicmatrix;
    /*
    FrameRGBD(std::string rgb_file, std::string depth_file)
    {
        if(load(rgb_file, depth_file))
        {
          // dx dy
        }
    }
  */

    FrameRGBD()
    {
        _intrinsicmatrix.fx = 517.3;
        _intrinsicmatrix.fy = 516.5;
        _intrinsicmatrix.ox = 318.6;
        _intrinsicmatrix.oy = 255.3;
        _intrinsicmatrix.depth_factor = 5000.0f;
    }
    bool load(std::string rgb_file, std::string depth_file);
    inline void convertRawDepthImage(const cv::Mat& input, cv::Mat& output, float scale);

    void buildPointCloud();
    void warpImg(const AffineTransform transform, const xyzPointCloud &refrence_pointcloud, FrameRGBD& warpedrgbd);
    void calculateImgDerivatives();
};

class RgbdPyramid
{
public :
    typedef boost::shared_ptr< dvo_framedata::RgbdPyramid > Ptr;

    int level_num;
    std::vector< FrameRGBD > levels;

    //RgbdPyramid():{}
    RgbdPyramid(int ii, FrameRGBD& finelevel);
    void buildPyr();
    void DownSampleInMatrix(const IntrinsicMatrix & IM, IntrinsicMatrix& downIM);

    //~FrameData();
};

}

#endif

