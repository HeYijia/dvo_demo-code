#include "CalWeight.h"

namespace dvo_core {

void WeightCaculation::calculateScale(const cv::Mat& errors)
{
    sigma_ = std::max( _tdistribution.TdistributionScaleEsitimator(errors), 0.001f);
}

/*
 *  computeWights ::
 *  calculate every pixel's wight and stack the result into a weightimg, this function used for debug, show the weight img
*/
void WeightCaculation::computeWights(const cv::Mat &residuals, cv::Mat &weights)
{
    weights.create(residuals.size(), residuals.type());
    float* weight_ptr = weights.ptr<float>();
    const float* residuals_ptr = residuals.ptr<float>();

    for(int idx=0; idx < residuals.size().area(); ++idx, ++residuals_ptr, ++weight_ptr)
    {
        if(std::isfinite( *residuals_ptr) )
        {
            *weight_ptr = _tdistribution.WeightVaule( * residuals_ptr  / sigma_);
        }else
        {
            *weight_ptr = 0.0f;
        }
    }
}
/*
 * computeWight: cacutlate one pixel's weight
 *         this function is called when compute the Matrix  J_idx * w_idx * J_idx
*/
float  WeightCaculation::computeWight(const float residual)
{
    return _tdistribution.WeightVaule( residual  / sigma_);
}
/*
 *    T distribution
*/
inline float InvSqrt(float x )
{
    // this function is faster 4.0 than the 1/std::sqrt()
    float xhalf = 0.5f*x;
    int i = *(int*)&x;                // get bits for floating vaule
    i = 0x5f375a86 - (i>>1); // gives inital guess y0.   what the fuck? ghost data 0x5f375a86
    x = *(float*)&i;                  // convert bits back to float
    x = x*(1.5f - xhalf*x*x);  // Newton step, repeating increases accuracy
    x = x*(1.5f - xhalf*x*x);  // Newton step, repeating increases accuracy
    x = x*(1.5f - xhalf*x*x);  // Newton step, repeating increases accuracy
    return x;
}

float Tdistribution::TdistributionScaleEsitimator(const cv::Mat& errors)
{
    float inital_lamda =1.0f / (inital_sigma_ *  inital_sigma_);
    float lamda = inital_lamda;
    float data_cnt = 0.0f;
    do
    {
        inital_lamda = lamda;
        data_cnt = 0.0f;
        lamda = 0.0f;
        const float *errors_ptr = errors.ptr<float>();

        for(int idx=0; idx< errors.size().area(); idx++, errors_ptr++)
        {
            const float err_data = *errors_ptr;
            if(std::isfinite(err_data))
            {
                float data = err_data * err_data;
                data_cnt += 1.0f ;
                lamda += data * ( (dof_ + 1.0f) / (dof_ + data * inital_lamda) );
            }
        }
        lamda /= data_cnt;
        lamda = 1.0f / lamda;
    }while(std::abs( lamda - inital_lamda ) > 1e-3);

    return InvSqrt(lamda);
}

float Tdistribution::WeightVaule(const float &x)
{
    return ((dof_ + 1.0f) / (dof_ + (x * x) ) );
}

}

