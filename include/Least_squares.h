#ifndef LEAST_SQUARES_H
#define LEAST_SQUARES_H

#include <Eigen/Core>
#include<Eigen/Dense>

class normalLeastSquares
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Eigen::Matrix<float,6,6> A;
    Eigen::Matrix<float,6,1> b;

    double error;
    int vaild_constraints;

    normalLeastSquares()
    {
        A.setZero();
        b.setZero();
        error = 0.0;
        vaild_constraints = 0;
    }

    void update(const Eigen::Matrix<float,6,1>& J, const float& res, const float& weight = 1.0f);
    void finish();
    void slove(Eigen::Matrix<float, 6,1>& x);

};
#endif // LEAST_SQUARES_H
