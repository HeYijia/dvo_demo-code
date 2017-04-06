#include "rgbdread.h"
#include "FrameData.h"
#include "viewer.h"
#include "TransformEstimate.h"

#include <Eigen/Core>
#include <Eigen/Dense>

int main(int argc, char** argv)
{
    std::string filepath = "/home/hyj/datasets/freiburg1_desk/";
    FileRead fileread(filepath+"assoc.txt");
    std::vector< rgbdPairs > allpairs;
    fileread.readAllEntries(allpairs);
    dvo_framedata::FrameRGBD current,reference,warp_test;
    dvo_framedata::RgbdPyramid::Ptr refPyramid,currentPyramid;


    Eigen::Affine3d trajectory,result, trajectory_showpclpoint;
    GroundtruthPose gt;

    dvo_viewer::viewer pointsview;

    std:: ostream* trajectory_out;
    trajectory_out = new std::ofstream("result.txt");
    for(std::vector< rgbdPairs >::iterator it= allpairs.begin(); it != allpairs.end(); ++it)
    {
        refPyramid = currentPyramid;

        current.load(filepath+ it->rgbfile, filepath+ it->depthfile);
        currentPyramid = dvo_framedata::RgbdPyramid::Ptr(new dvo_framedata::RgbdPyramid(4, current) );
        currentPyramid->buildPyr();

        if(!refPyramid)
        {
            bool flag = true;
            FindCloseposeInGroundtruth(filepath + "groundtruth.txt" , gt ,it-> rgbTimestamp,trajectory);
            pointsview.init(currentPyramid->levels[ 0 ]);
            continue;
        }

         //std::cout << refPyramid->levels[0].frameID<<" currentPyramid: "<<currentPyramid->levels[0].frameID<<std::endl;

        reference.load(filepath+ (it-1)->rgbfile, filepath+ (it-1)->depthfile);
        refPyramid = dvo_framedata::RgbdPyramid::Ptr(new dvo_framedata::RgbdPyramid(4, reference) );
        refPyramid ->buildPyr();

        dvo_core::DenseTrack densetrack(currentPyramid,refPyramid);
        result.setIdentity();
        densetrack.match(currentPyramid,refPyramid,result);

        trajectory = trajectory * result ;

        pointsview.showcloud(currentPyramid->levels[ 0 ],result);

        Eigen::Quaterniond q(trajectory.rotation()) ;
        *trajectory_out <<std::fixed <<it->rgbTimestamp<<" "
                          << trajectory.translation()(0)<<" "
                           << trajectory.translation()(1)<<" "
                           << trajectory.translation()(2)<<" "
                           << q.x()<<" "
                            << q.y()<<" "
                            << q.z()<<" "
                            << q.w()<<std::endl;

        std::cout<<std::fixed<< it->rgbTimestamp<<" "
                << trajectory.translation()(0)<<" "
                 << trajectory.translation()(1)<<" "
                 << trajectory.translation()(2)<<" "
                 << q.x()<<" "
                  << q.y()<<" "
                  << q.z()<<" "
                  << q.w()<<std::endl;

    }

    return 0;
}
