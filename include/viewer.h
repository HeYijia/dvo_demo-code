#ifndef VIEWER_H
#define VIEWER_H

#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

#include <FrameData.h>
#include <Eigen/Core>

namespace dvo_viewer
{

pcl::PointCloud< pcl::PointXYZRGBA>::Ptr img2pcl(dvo_framedata::FrameRGBD & frame);
pcl::PointCloud< pcl::PointXYZRGBA>::Ptr JoinPointCloud(pcl::PointCloud< pcl::PointXYZRGBA>::Ptr oldpointclouds,dvo_framedata::FrameRGBD & newframe,Eigen::Affine3d transform);

class viewer
{
public:
    pcl::PointCloud< pcl::PointXYZRGBA>::Ptr oldpoints;
    Eigen::Affine3d transform_cur2world;
    pcl::visualization::CloudViewer show;

    viewer();
    void init(dvo_framedata::FrameRGBD & frame);
    void showcloud(dvo_framedata::FrameRGBD &frame, Eigen::Affine3d transform);
};

}

#endif // VIEWER_H
