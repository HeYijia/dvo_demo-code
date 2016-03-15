#ifndef TRANSFORMESTIMATE_H
#define TRANSFORMESTIMATE_H
#include"CalWeight.h"
#include"Least_squares.h"
namespace dvo_core {
class DenseTrack
{
public:
    DenseTrack(dvo_framedata::RgbdPyramid::Ptr cPyr, dvo_framedata::RgbdPyramid::Ptr rPyr);
    void computeJw(const Eigen::Vector4f p, Eigen::Matrix<float,2,6>& jw);
    void computeLSEquationsGeneric(dvo_framedata::FrameRGBD& ref, dvo_framedata::FrameRGBD& cur, const AffineTransform& transform);
    void LSEquationreduction(const cv::Mat &residuals, const cv::Mat& Jix, const cv::Mat& Jiy, const dvo_framedata::FrameRGBD::xyzPointCloud& points);

    void match(dvo_framedata::RgbdPyramid::Ptr cPyr, dvo_framedata::RgbdPyramid::Ptr rPyr, Eigen::Affine3d &transformation);

private:
    WeightCaculation weights_;
    normalLeastSquares ls;
    dvo_framedata::RgbdPyramid::Ptr currentPyr;
    dvo_framedata::RgbdPyramid::Ptr referencePyr;

};

}

#endif
