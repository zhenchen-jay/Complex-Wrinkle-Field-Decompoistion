#pragma once

#include "MeshLib/Mesh.h"
#include "CommonTools.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

void computeEdgeMatrix(const Mesh& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& edgeWeight, const int nverts, Eigen::SparseMatrix<double>& A);
void computeEdgeMatrixGivenMag(const Mesh& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& edgeWeight, const int nverts, Eigen::SparseMatrix<double>& A);

void roundZvalsFromEdgeOmega(const Mesh& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& edgeWeight, const Eigen::VectorXd& vertArea, int nverts, std::vector<std::complex<double>>& zvals);
void roundZvalsFromEdgeOmegaVertexMag(const Mesh& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& edgeWeight, const Eigen::VectorXd& vertArea, int nverts, std::vector<std::complex<double>>& zvals);



