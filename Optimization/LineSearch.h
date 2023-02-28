#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

double armijoLineSearch(const Eigen::VectorXd& x0, const Eigen::VectorXd& grad, const Eigen::VectorXd& dir, const std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool isProj)> objFunc, double r0 = 1, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> bbxProj = nullptr);