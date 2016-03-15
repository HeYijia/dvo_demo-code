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

    /*
    reference.load(filepath + "rgb/1311868164.363181.png", filepath+ "depth/1311868164.373557.png");
    current.load(filepath + "rgb/1311868164.399026.png", filepath+ "depth/1311868164.407784.png");

    Eigen::Quaternionf q1(-0.4095,0.6529,-0.5483,0.3248);
    Eigen::Vector3f t1(-0.1546, -1.4445, 1.4773);
    Eigen::Matrix3f r1(q1.toRotationMatrix());

    Eigen::Quaternionf q2(-0.4080,0.6564,-0.5474,0.3210);
    Eigen::Vector3f t2(-0.1578, -1.4458, 1.4770);
    Eigen::Matrix3f r2(q2.toRotationMatrix());
    Eigen::Affine3f cr; // transform point to c from r
    cr.matrix().topLeftCorner<3,3>() = r2.transpose() * r1;
    cr.translation() =  r2.transpose() * (t1 - t2);
    std::cout << "transformation : " <<std::endl << cr.matrix() <<std::endl;
*/
    clock_t begin,end;
    double time_spend;
    begin = clock();

    end = clock();
    time_spend = (double)(end - begin)/CLOCKS_PER_SEC;
    std::cout << "GOT A FRAME , Time SPEND:"<<time_spend<<std::endl;


    Eigen::Affine3d trajectory,result, trajectory_showpclpoint;
    GroundtruthPose gt;
/*
      FindCloseposeInGroundtruth(filepath + "groundtruth.txt" , gt ,1311868164.363181 ,trajectory);
      Eigen::Quaterniond q(trajectory.rotation()) ;
    std::cout<< trajectory.translation() <<"  "<< q.x() <<"  "<<q.y()<<"  "<<q.z()<<"  "<<q.w()<<std::endl;
*/

/*
    refPyramid = dvo_framedata::RgbdPyramid::Ptr(new dvo_framedata::RgbdPyramid(3, reference) );
    refPyramid->buildPyr();

    currentPyramid = dvo_framedata::RgbdPyramid::Ptr(new dvo_framedata::RgbdPyramid(3, current) );
    currentPyramid->buildPyr();

    dvo_core::DenseTrack densetrack(refPyramid,currentPyramid);
    result.setIdentity();
    densetrack.match(currentPyramid,refPyramid,result);
    //std::cout << result.inverse().matrix()<<std::endl;
    //std::cout << result.matrix()<<std::endl;

    //refPyramid->levels[0].buildPointCloud();
    currentPyramid->levels[0].warpImg(result.inverse().cast<float>(), refPyramid->levels[0]._xyzclouds, warp_test);
    cv::Mat residuals =  warp_test._grey - refPyramid->levels[0]._grey;
    cv::imshow("level2",residuals);
    //cv::imshow("ss",refPyramid->levels[0]._dpt_16u);

    currentPyramid->levels[0].warpImg(cr, refPyramid->levels[0]._xyzclouds, warp_test);
    cv::Mat crresiduals =  warp_test._grey - refPyramid->levels[0]._grey;
    cv::imshow("cr",crresiduals);

    cv::waitKey(0);
*/

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
