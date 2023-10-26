#pragma once
#include "CommonTools.h"

void extractIsopoints(const Mesh& mesh, const Eigen::VectorXd& func, double isoVal, MatrixX& isoV);

double barycentric(double val1, double val2, double target);
bool crosses(double isoval, double val1, double val2, double& bary);
int extractIsoline(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXi& faceNeighbors,
                   const Eigen::VectorXd& func, double isoval, Eigen::MatrixXd& isoV, Eigen::MatrixXi& isoE);
int extractIsoline(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXi& faceNeighbors,
                   const Eigen::VectorXd& func, double isoval, Eigen::MatrixXd& isoV, Eigen::MatrixXi& isoE,
                   Eigen::MatrixXd& extendedV, Eigen::MatrixXi& extendedF);