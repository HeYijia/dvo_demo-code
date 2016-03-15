#ifndef RGBDREAD_H_
#define RGBDREAD_H_
/*
 *     rgbdPairs is used to read read rgb_file_name and depth_file_name from assoc.txt in the TUM rgbddata
 *     assoc.txt can be get by TUM rgbdtools
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Geometry>

struct GroundtruthPose
{
    double poinstionX;
    double poinstionY;
    double poinstionZ;
    double orientationX;
    double orientationY;
    double orientationZ;
    double orientationW;
    double poseTimestamp;

    friend std::istream& operator>>(std::istream &input , GroundtruthPose &gt)
    {
        input >> gt.poseTimestamp;
        input >> gt.poinstionX;
        input >> gt.poinstionY;
        input >> gt.poinstionZ;
        input >> gt.orientationX;
        input >> gt.orientationY;
        input >> gt.orientationZ;
        input >> gt.orientationW;
        return input;
    }

};
void FindCloseposeInGroundtruth(std::string filepath, GroundtruthPose& gt, const double time, Eigen::Affine3d& pose)
{
    std::ifstream file_stream(filepath.c_str());
    while(file_stream.good() && file_stream.peek() == '#')
    {
            file_stream.ignore(1024,'\n');
    }

    bool sucess = true;
    gt.poseTimestamp = 0.0;
    while(gt.poseTimestamp < time)
    {
        if(file_stream.good() && !file_stream.eof() )
        {
            file_stream >> gt;
        }
        else
        {
            sucess = false;
            break;
        }
    }
    if(sucess)
    {
        Eigen::Quaterniond rotation(gt.orientationW,gt.orientationX,gt.orientationY,gt.orientationZ);
        Eigen::Vector3d t(gt.poinstionX,gt.poinstionY,gt.poinstionZ);
        pose = Eigen::Affine3d::Identity();
        pose = rotation * pose;
        pose.translation() = t;
    }

}

struct rgbdPairs
{
    std::string rgbfile;
    std::string depthfile;
    double rgbTimestamp;
    double depthTimestamp;
    rgbdPairs(){}
    friend std::istream& operator>>(std::istream &input ,rgbdPairs &rdp)
    {
        input >> rdp.rgbTimestamp>>rdp.rgbfile>>rdp.depthTimestamp>>rdp.depthfile;
        return input;
    }

};

class FileRead
{
public:

    FileRead(std::string  file):
        _file_stream(file.c_str())
    {
        skipcomment();
    }

    virtual ~FileRead()
    {
        _file_stream.close();
    }

    inline void skipcomment()
    {
        while(_file_stream.good() && _file_stream.peek() == '#')
        {
                _file_stream.ignore(1024,'\n');
        }
    }

    bool next()
    {
        if(_file_stream.good() && !_file_stream.eof() )
        {
            _file_stream >> _rdp;
            return true;
        }
        return false;
    }

    void readAllEntries(std::vector< rgbdPairs >& allpairs)
    {
        while(next())
        {
            allpairs.push_back(_rdp);
        }
    }
private:
    //std::string _filename;
    std::ifstream _file_stream;
    rgbdPairs _rdp;
};
#endif
