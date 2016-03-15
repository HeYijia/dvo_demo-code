#include "FrameData.h"
#include "imageprocess.h"

void saveMatToCsv(cv::Mat &data, std::string filename)
{
    std::ofstream outputFile(filename.c_str());
    outputFile << cv::format(data,"CSV")<<std::endl;
    outputFile.close();
}
const static Eigen::IOFormat CSVformat(Eigen::StreamPrecision,Eigen::DontAlignCols, "," , "\n" );
void saveEigenToCsv(Eigen::MatrixXf matrix, std::string filename)
{
    std::ofstream outputFile(filename.c_str(),std::fstream::app);
    outputFile << matrix.format(CSVformat)<<std::endl;
    outputFile.close();
}

namespace dvo_framedata {

bool FrameRGBD::load(std::string rgb_file, std::string depth_file)
{
    cv::Mat rgb, grey, depth;
    bool rgb_available=false;

   rgb = cv::imread(rgb_file, 1);
   depth = cv::imread(depth_file, -1);
   if(rgb.empty() || depth.empty())
   {
       std::cout << "rgb or depth file is empty"<<std::endl;
       return false;
   }

   if(rgb.type() != CV_32FC1)
   {
       if(rgb.type() == CV_8UC3)
       {
           cv::cvtColor(rgb, grey, CV_BGR2GRAY);
           rgb_available = true;
       }
       else
       {
            grey = rgb;
       }
       grey.convertTo(_intensity, CV_32F);
   }
   else
   {
        _intensity = rgb;
   }

   if(depth.type() != CV_32FC1)
   {
       _dpt_16u = depth;
        convertRawDepthImage(depth, _dpt, 1.0f / 5000.0f);
   }
   else
   {
        _dpt = depth;
   }

   //if(rgb_available)rgb.convertTo(_rgb, CV_32FC3);
   _rgb = rgb;
   _grey = grey;
   frameID = rgb_file;
   return true;
}

inline void FrameRGBD::convertRawDepthImage(const cv::Mat& input, cv::Mat& output, float scale)
{
    output.create(input.rows, input.cols, CV_32FC1);
    float a = 0.0f;
    const unsigned short* input_ptr = input.ptr<unsigned short>();
    float* output_ptr = output.ptr<float>();
    for(int idx = 0; idx < input.size().area(); idx++, input_ptr++, output_ptr++)
    {
        if(*input_ptr == 0)
        {
            *output_ptr = std::numeric_limits<float>::quiet_NaN();
        }
        else
        {
            *output_ptr = ((float) *input_ptr) * scale;
        }
    }
}

/*
 * buildPointCloud:  project img coordinate (x,y) to the  camera coordinate (X,Y,Z).
 */
void FrameRGBD::buildPointCloud()
{
    // build the pointcloud when the _dpt img and the intrinsicMatix has load
    _xyzclouds.resize(Eigen::NoChange, _dpt.cols * _dpt.rows);  // _xyzclouds size is : 4 rows , w*h clos
    int idx =0;
    for(int y = 0; y < _dpt.rows; ++y)
    {
        for(int x = 0; x < _dpt.cols; ++x,++idx)
        {
            float Z = _dpt.ptr<float>(y)[x];

            _xyzclouds(0, idx) = (x - _intrinsicmatrix.ox)/ _intrinsicmatrix.fx * Z;
            _xyzclouds(1, idx) = (y - _intrinsicmatrix.oy)/ _intrinsicmatrix.fy * Z;
            _xyzclouds(2, idx) = Z;
            _xyzclouds(3, idx) = 1.0f;
        }
    }
}

void FrameRGBD::warpImg(const AffineTransform transform, const xyzPointCloud& refrence_pointcloud, FrameRGBD &warpedrgbd)
{
    cv::Mat warped_intensity(_intensity.size(), _intensity.type());
    cv::Mat warped_grey(_grey.size(), _grey.type());   //  debug
    cv::Mat warped_dpt(_dpt.size(), _dpt.type());         // warped_depth can be used to count the geometry error to improve the dvo algorithm.

    xyzPointCloud transformedpointcloud;
   // step 1 : transfrom the point XYZ to the current camera coordinate.  paper fomular: g(G,p)
    transformedpointcloud.resize(Eigen::NoChange, _dpt.cols * _dpt.rows);
    transformedpointcloud= transform * refrence_pointcloud;

    int idx = 0;
    for(int y = 0; y < _intensity.rows; ++y)
    {
        uchar a = 0;
        float b = 0;
        for(int x = 0; x < _intensity.cols; ++x,++idx)
        {
            Eigen::Vector4f transformedXYZ1 = transformedpointcloud.col( idx );

            // step 2 : project transformedXYZ1 to the current image coordinate xy
            if( ! std::isfinite( transformedXYZ1(2) ) )
            {
                warped_intensity.ptr<float>(y)[x] = std::numeric_limits<float>::quiet_NaN();
                warped_grey.ptr<uchar>(y)[x] = std::numeric_limits<uchar>::quiet_NaN();
                warped_dpt.ptr<float>(y)[x] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            float x_projected = (float) (transformedXYZ1(0) * _intrinsicmatrix.fx / transformedXYZ1(2) + _intrinsicmatrix.ox);
            float y_projected = (float)(transformedXYZ1(1) * _intrinsicmatrix.fy / transformedXYZ1(2) + _intrinsicmatrix.oy);

            //step 3 : use the bilinear interpolate algorithm to get the warped img
            if(x_projected >=0 && x_projected < _intensity.cols && y_projected >=0 && y_projected < _intensity.rows) // point in image ?
            {
                //warped_grey.ptr<uchar>(y)[x] = bilinear<uchar>( _grey, x_projected , y_projected);
                warped_grey.ptr<uchar>(y)[x] = bilinearWithdepth<uchar>( _grey, _dpt ,x_projected , y_projected, transformedXYZ1(2));
                warped_intensity.ptr<float>(y)[x] = bilinearWithdepth<float>(_intensity, _dpt ,x_projected , y_projected, transformedXYZ1(2));
                a = warped_grey.ptr<uchar>(y)[x] ;
                b = warped_intensity.ptr<float>(y)[x];
                warped_dpt.ptr<float>(y)[x] = transformedXYZ1(2) ;
            }else
            {
                warped_intensity.ptr<float>(y)[x] = std::numeric_limits<float>::quiet_NaN();
                warped_dpt.ptr<float>(y)[x] = std::numeric_limits<float>::quiet_NaN();
            }

        }
    }
    warpedrgbd._grey = warped_grey;
    warpedrgbd._intensity = warped_intensity;
    warpedrgbd._dpt = warped_dpt;
    /*
    cv::imshow("warped_intensity",warped_grey);
    cv::imshow("warp_dpt",warped_dpt);
    cv::waitKey(0);
    */
}

void FrameRGBD::calculateImgDerivatives()
{
    calculateDerivativeX<float>(_intensity,_intensity_dx);
    calculateDerivativeY<float>(_intensity,_intensity_dy);
}

/*
void dvo_framedata::FrameRGBD::warpImg(const AffineTransform transform, FrameRGBD &warpedrgbd)
{
    cv::Mat warped_intensity(_intensity.size(), _intensity.type());
    cv::Mat warped_grey(_grey.size(), _grey.type());
    cv::Mat warped_dpt(_dpt.size(), _dpt.type());
    // step 1:  projection
    Eigen::Vector4f XYZ1 ;
    Eigen::Vector4f transformedXYZ1;

    const float* dpt_ptr = _dpt.ptr<float>();
    uchar* warp_ptr = warped_grey.ptr<uchar>();
    float* warpdpt_ptr = warped_dpt.ptr<float>();

    for(int idx = 0; idx < _dpt.size().area(); idx++, dpt_ptr++, warp_ptr++,warpdpt_ptr++)
    {
        // step 1 : project xy to the XYZ current camera coordinate.  paper fomular : inv_pi ( x, z)
        float Z = *dpt_ptr;
        int x= idx%480;
        int y = idx/480;
        XYZ1(0) = (x - _intrinsicmatrix.ox)/ _intrinsicmatrix.fx * Z;
        XYZ1(1) = (y - _intrinsicmatrix.oy)/ _intrinsicmatrix.fy * Z;
        XYZ1(2) = Z;
        XYZ1(3) = 1.0;

        transformedXYZ1= transform * XYZ1;

        if( ! std::isfinite( transformedXYZ1(2) ) )
        {
            *warp_ptr= std::numeric_limits<uchar>::quiet_NaN();
            *warpdpt_ptr= std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        float x_projected = (float) (transformedXYZ1(0) * _intrinsicmatrix.fx / transformedXYZ1(2) + _intrinsicmatrix.ox);
        float y_projected = (float)(transformedXYZ1(1) * _intrinsicmatrix.fy / transformedXYZ1(2) + _intrinsicmatrix.oy);

        //step 4 : use the bilinear interpolate algorithm to get the warped img
        if(x_projected >=0 && x_projected < _intensity.cols && y_projected >=0 && y_projected < _intensity.rows) // point in image ?
        {
            //warped_grey.ptr<uchar>(y)[x] = bilinear<uchar>( _grey, x_projected , y_projected);
            *warp_ptr = bilinearWithdepth<uchar>( _grey, _dpt ,x_projected , y_projected, transformedXYZ1(2));
            *warpdpt_ptr = transformedXYZ1(2) ;
        }else
        {
            *warp_ptr= std::numeric_limits<uchar>::quiet_NaN();
            *warpdpt_ptr= std::numeric_limits<float>::quiet_NaN();
        }
    }
    warpedrgbd._grey = warped_grey;
}
*/
/*
 *    *  *       Pyramid   Class     *  *
*/
// init Pyramid: put the finest img into the pyramid vector, levels[0] = finelevel
RgbdPyramid::RgbdPyramid(int ii, FrameRGBD& finelevel):
    level_num(ii),levels(1,finelevel)
{

}

void RgbdPyramid::DownSampleInMatrix(const IntrinsicMatrix & IM, IntrinsicMatrix& downIM)
{
    downIM.fx = IM.fx / 2.0f;
    downIM.fy = IM.fy / 2.0f;
    downIM.ox = IM.ox / 2.0f;
    downIM.oy = IM.oy / 2.0f;
}
void RgbdPyramid::buildPyr()
{
    for(int i = 0; i < level_num-1; i++)
    {
        FrameRGBD temp_level;
        pyrDownMean<uchar>( levels[i]._grey ,  temp_level._grey);
        pyrDownMedian<ushort>(levels[i]._dpt_16u ,   temp_level._dpt_16u);
        pyrDownMedian<float>( levels[i]._dpt ,  temp_level._dpt);
        pyrDownMean<float>( levels[i]._intensity ,  temp_level._intensity);
        DownSampleInMatrix(levels[i]._intrinsicmatrix,temp_level._intrinsicmatrix);
        levels.push_back(temp_level);
    }
}

}
