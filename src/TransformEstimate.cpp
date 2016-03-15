#include "TransformEstimate.h"
#include "sophus_src/sophus/se3.h"


namespace dvo_core {

DenseTrack::DenseTrack(dvo_framedata::RgbdPyramid::Ptr cPyr, dvo_framedata::RgbdPyramid::Ptr rPyr):
    currentPyr(cPyr),referencePyr(rPyr)
{

}

/*
 *  computeJw : compute the jacobian of warp function
 *  notice that
 *          jw_truth(0,i) = jw(0,i)* IntrinsicMatrix.fx
 *          jw_truth(1,i) = jw(1,i)* IntrinsicMatrix.fy
 *  considering the efficent, we multiply the fx, fy at the  J = Ji * Jw.
 *
*/
void DenseTrack:: computeJw(const Eigen::Vector4f p, Eigen::Matrix<float,2,6>& jw)
{
    float z = 1.0f / p(2);
    float z_sqr = 1.0f / ( p(2) * p(2));

    jw(0,0) =   z;
    jw(0,1) = 0.0f;
    jw(0,2) = - p(0) * z_sqr;
    jw(0,3) = jw(0,2) * p(1) ;
    jw(0,4) = 1.0f - jw(0,2) * p(0);
    jw(0,5) = - p(1) * z;

    jw(1,0) = 0.0f;
    jw(1,1) = z;
    jw(1,2) = - p(1)* z_sqr;
    jw(1,3) = -1.0f + jw(1,2) * p(1);
    jw(1,4) = -jw(0,3);
    jw(1,5) =  p(0) * z;

}

/*
 * LSEquationreduce :  accumlate all the pixels jacabian Ji,Jw to Least square coff A,b.
 *        Ji is jacobian of img, Ji is <Ji_dx, Ji_dy> 1x2 vector
 *        Jw is jacobian of warp funtion. jw 2x6 vector
 *        J_idx = Ji * Jw,
 *        Ax = b
 *        A =  sum ( J_idx * J_idx.transpose() * weight_idx ) .   A is 6x6 matrix
 *        b =  sum (- J_idx * res * weight_idx).     b is 6x1 vector
 */
void DenseTrack::LSEquationreduction(const cv::Mat& residuals, const cv::Mat &Jix, const cv::Mat &Jiy, const dvo_framedata::FrameRGBD::xyzPointCloud &points)
{
    Eigen::Matrix<float,1,2> Ji;
    Eigen::Matrix<float,2,6> Jw;

    const float* jix_ptr = Jix.ptr<float>();
    const float* jiy_ptr = Jiy.ptr<float>();
    const float* residual_ptr = residuals.ptr<float>();

    int size = residuals.size().area();
    normalLeastSquares ls_temp;
    for ( int idx = 0; idx < size; ++idx, ++jix_ptr, ++ jiy_ptr,++residual_ptr)
    {
        if ( !std::isfinite(*jix_ptr) || !std::isfinite(*jiy_ptr) || !std::isfinite(*residual_ptr) ) continue;
        Ji << *jix_ptr, *jiy_ptr;

        Eigen::Vector4f p = points.col(idx);
        computeJw(p, Jw);

        //saveEigenToCsv(Jw,"/home/hyj/Jw.csv");   // hyj comment

        ls_temp.update(Ji * Jw, *residual_ptr, weights_.computeWight(*residual_ptr));
    }

    ls_temp.finish();
    ls = ls_temp;
    //std::cout <<ls.A<<std::endl;
    //std::cout<<"DEBUG break point"<<std::endl;
}

void DenseTrack::computeLSEquationsGeneric(dvo_framedata::FrameRGBD& ref,  dvo_framedata::FrameRGBD& cur,const AffineTransform& transform)
{
    //
    dvo_framedata::FrameRGBD cur_warped;
    cv::Mat residuals, ref_dx, ref_dy;

    // step 1. compute jacobian of img Jix,Jiy and multiply with fx ,fy
    // notice that compute Jw not multiply the _intrinsicmatrix.fx , _intrinsicmatrix.fy
    ref.calculateImgDerivatives();
    ref_dx = ref._intensity_dx * ref._intrinsicmatrix.fx ;
    ref_dy = ref._intensity_dy * ref._intrinsicmatrix.fy ;

    // step 2. compute jacobian of warp function Jw
    //ref.buildPointCloud();  //just need build once
    cur.warpImg(transform,ref._xyzclouds, cur_warped);
    residuals = cur_warped._intensity - ref._intensity ;

    //saveMatToCsv(residuals,"/home/hyj/hyj_residuals.csv");
    //saveMatToCsv(ref_dy,"/home/hyj/hyj_ref_dy.csv");
    /*
    std::cout <<ref.frameID<<" currentPyramid: "<<cur.frameID<<std::endl;
    cv::imshow("res",debug);
    cv::imshow("debug_res",debug_residuals);
    cv::imshow("ref",ref._grey);
    cv::imshow("cur",cur._grey);
    cv::waitKey(0);
*/
    weights_.calculateScale(residuals);

    //saveEigenToCsv(ref._xyzclouds,"/home/hyj/xyzclouds.csv");

    LSEquationreduction(residuals , ref_dx, ref_dy, ref._xyzclouds);

    //Eigen::Matrix<float,6,1> x;
    //ls.slove(x);
    //Eigen::Affine3d t;
    //Sophus::SE3 inc(t.rotation(), t.translation());
    //Sophus::SE3 inc= Sophus::SE3::exp(x.cast<double>());
    //std::cout<< inc.matrix() <<std::endl;

}

void DenseTrack::match(dvo_framedata::RgbdPyramid::Ptr cPyr, dvo_framedata::RgbdPyramid::Ptr rPyr, Eigen::Affine3d& transformation)
{
    Sophus::SE3 inc(transformation.rotation(), transformation.translation());
    AffineTransform estimate(AffineTransform::Identity()), prev_estimate(AffineTransform::Identity());
    bool accept = true;
    int iteration = 0;
/*
    cv::imshow("ref",rPyr->levels[ 0 ]._grey);
    cv::imshow("cur",cPyr->levels[ 0 ]._grey);
    cv::waitKey(0);
    std::cout <<rPyr->levels[0].frameID<<" currentPyramid: "<<cPyr->levels[0].frameID<<std::endl;
*/
    int coarse_lev_num = cPyr->level_num - 1;
    for( int lev_idx = coarse_lev_num; lev_idx >=1; -- lev_idx)
    {
        iteration = 0;
        dvo_framedata::FrameRGBD& ref = rPyr->levels[ lev_idx ] ;
        dvo_framedata::FrameRGBD& cur = cPyr->levels[ lev_idx ] ;
        ref.buildPointCloud();

        Eigen::Matrix<float,6,1> x;
        float prev_errors = 1.0f;
        float errordiff =0.0f;
        do
        {
            estimate = inc.matrix().cast<float>() * estimate.matrix();
            computeLSEquationsGeneric(ref,cur,estimate);

            errordiff =prev_errors - ls.error ;
            accept = errordiff > 0.0f ;
            prev_errors = ls.error;
            // if we get a worse result , we use the prev_estimate and try our luck on the next pyramid level
            if( !accept )
            {
                   estimate = prev_estimate;
                   inc = Sophus::SE3::exp(Eigen::Matrix<double,6,1>::Zero());
                   break;
            }
            prev_estimate = estimate ;
            ls.slove(x);
            inc = Sophus::SE3::exp( x.cast< double >() );
            iteration++ ;
        }while( accept && errordiff > 5e-7 && iteration < 100);
    }

    if( iteration >= 100 )
    {
         estimate = inc.matrix().cast<float>() * estimate.matrix();
    }

    // estimate  is transfrom ref coodinate into the cur coodinate Tcr, used to warp cur img to ref img  by inverse method
    // transformation is tranfrom cur coodinate into ref coodinate Trc.   Twc = Twr * Trc ;   w is the world coodinate.
    transformation = estimate.inverse().cast<double>();
}

}

