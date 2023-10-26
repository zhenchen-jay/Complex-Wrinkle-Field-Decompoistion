#pragma once

#include "LineSearch.h"

void ProjectedNewtonSolver(
    std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc,
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> findMaxStep, const Eigen::VectorXd& lx,
    const Eigen::VectorXd& ux, Eigen::VectorXd& x0, int numIter = 1000, double gradTol = 1e-14, double xTol = 0,
    double fTol = 0, bool displayInfo = false);
// implementation of the projected newton mentioned in the paper: Tackling box-constrained optimization via a new
// projected quasi-Newton approach (https://www.cs.utexas.edu/~inderjit/public_papers/pqnj_sisc10.pdf)

void testProjectedNewton();