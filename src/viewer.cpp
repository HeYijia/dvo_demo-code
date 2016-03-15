#include <viewer.h>

namespace dvo_viewer {

pcl::PointCloud< pcl::PointXYZRGBA>::Ptr img2pcl(dvo_framedata::FrameRGBD & frame)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudPtr(new pcl::PointCloud<pcl::PointXYZRGBA>);

    cv::Mat rgb = frame._rgb;
    cv::Mat dpt = frame._dpt_16u;

    cloudPtr->width = dpt.cols;
    cloudPtr->height = dpt.rows;
    cloudPtr->is_dense = false;
    cloudPtr->points.resize(cloudPtr->width * cloudPtr->height);

    for(int v = 0; v < dpt.rows; v++)
    {
        const uchar *rgbptr = rgb.ptr<uchar>(v);
        for(int u = 0; u< dpt.cols; u++)
        {
            pcl::PointXYZRGBA xyzrgb;
            // rgb
            const uchar *pixel = rgbptr;
            xyzrgb.b = pixel[0];
            xyzrgb.g = pixel[1];
            xyzrgb.r = pixel[2];
            rgbptr += 3;
            // depth
            float Z = dpt.ptr<ushort>(v)[u]/ frame._intrinsicmatrix.depth_factor;
            float X = (u - frame._intrinsicmatrix.ox)*Z/frame._intrinsicmatrix.fx;
            float Y = (v - frame._intrinsicmatrix.oy)*Z/frame._intrinsicmatrix.fy;

            if(Z == 0.0)
            {
                xyzrgb.x = xyzrgb.y = xyzrgb.z = std::numeric_limits<float>::quiet_NaN();
            }
            else
            {
                xyzrgb.z = Z;
                xyzrgb.x = X;
                xyzrgb.y = Y;
            }
            cloudPtr->at(u,v) = xyzrgb;
        }
    }
    return cloudPtr;
}


pcl::PointCloud< pcl::PointXYZRGBA>::Ptr JoinPointCloud(pcl::PointCloud< pcl::PointXYZRGBA>::Ptr oldpointclouds,dvo_framedata::FrameRGBD & newframe,Eigen::Affine3d transform)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr newcloud = img2pcl(newframe);

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::transformPointCloud(*newcloud , *output, transform.matrix());

    *oldpointclouds += *output;

    static pcl::VoxelGrid<pcl::PointXYZRGBA> voxel;
    voxel.setLeafSize(0.01f,0.01f,0.01f);
    voxel.setInputCloud(oldpointclouds);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr showout(new pcl::PointCloud<pcl::PointXYZRGBA>);
    voxel.filter(*showout);
    return showout;
}

viewer::viewer():show("show point clouds")
{
    //pcl::visualization::CloudViewer viewer("show point clouds");
}
void viewer::init(dvo_framedata::FrameRGBD & frame)
{
    transform_cur2world.setIdentity();
    oldpoints = img2pcl(frame);
}

void viewer::showcloud(dvo_framedata::FrameRGBD & frame, Eigen::Affine3d transform)
{
    transform_cur2world = transform_cur2world * transform;
    oldpoints =dvo_viewer:: JoinPointCloud(oldpoints,frame, transform_cur2world);
    show.showCloud(oldpoints);
}

}
