#include "LineSearch.h"

double armijoLineSearch(const Eigen::VectorXd& x0, const Eigen::VectorXd& grad, const Eigen::VectorXd& dir, const std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool isProj)> objFunc, double r0, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> bbxProj)
{
    const double c = 0.2;
    const double rho = 0.5;
    double r = r0;

    Eigen::VectorXd x = x0 + r * dir;
    if(bbxProj)
        x = bbxProj(x);
    double f = objFunc(x, nullptr, nullptr, false);
    double f0 = objFunc(x0, nullptr, nullptr, false);
    const double cache = c * grad.dot(dir);

    while (f > f0 + r * cache) {
        r *= rho;
        x = x0 + r * dir;
        if (bbxProj)
            x = bbxProj(x);
        f = objFunc(x, nullptr, nullptr, false);
    }

    return r;
}