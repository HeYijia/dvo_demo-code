#include"Least_squares.h"

#include<iostream>
void normalLeastSquares::update(const Eigen::Matrix<float, 6, 1> &J, const float &res, const float &weight)
{
    float factor = weight / ( 255.0 * 255.0);

    //std::cout<<"J * J : "<<std::endl<<  J * J.transpose()<<std::endl;
    A += J * J.transpose() * factor;
    b  -= J * res * factor;

    error += res*res*factor;

    vaild_constraints += 1;
}

void normalLeastSquares::finish()

{
    A /= (float)vaild_constraints;
    b /= (float)vaild_constraints;
    error /= (float)vaild_constraints;
}

void normalLeastSquares::slove(Eigen::Matrix<float, 6, 1> &x)
{
    x = A.ldlt().solve(b);
}
