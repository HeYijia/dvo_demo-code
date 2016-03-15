#ifndef CALWEIGHT_H
#define CALWEIGHT_H

#include "FrameData.h"
#include <Eigen/Geometry>

namespace dvo_core {

class Tdistribution
{
public:
    float dof_;   // degree of freedom, can find in the paper
    float inital_sigma_;

    Tdistribution():dof_(5.0f),inital_sigma_(5.0)
    {

    }
    void configparam(float x)
    {
        dof_ = x;
    }

    float TdistributionScaleEsitimator(const cv::Mat& errors);
    float WeightVaule(const float& x);

};

class WeightCaculation
{
public:
    WeightCaculation():sigma_(1.0f){}
    void calculateScale(const cv::Mat& errors);
    void computeWights(const cv::Mat& residuals, cv::Mat & weights);
    float computeWight(const float residual);
private:
    float sigma_;
    Tdistribution _tdistribution;
};

}

#endif // CALWEIGHT_H
