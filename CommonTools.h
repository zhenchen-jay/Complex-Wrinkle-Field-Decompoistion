#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "MeshLib/Mesh.h"

#ifndef GRAIN_SIZE
#define GRAIN_SIZE 10
#endif

struct QuadraturePoints
{
    double u;
    double v;
    double weight;
};

std::vector<QuadraturePoints> buildQuadraturePoints(int order); // this is based one the paper: http://lsec.cc.ac.cn/~tcui/myinfo/paper/quad.pdf and the corresponding source codes: http://lsec.cc.ac.cn/phg/download.htm (quad.c)

Eigen::Vector3d computeHatWeight(double u, double v);

Eigen::MatrixXd SPDProjection(Eigen::MatrixXd A);

Eigen::VectorXd vertexVec2IntrinsicVec(const Eigen::MatrixXd& v, const Mesh& mesh);

Eigen::VectorXd faceVec2IntrinsicEdgeVec(const Eigen::MatrixXd& v, const Mesh& mesh);
Eigen::MatrixXd intrinsicEdgeVec2FaceVec(const Eigen::VectorXd& v, const Mesh& mesh);

Eigen::VectorXd getFaceArea(const Mesh& mesh);
Eigen::VectorXd getEdgeArea(const Mesh& mesh);
Eigen::VectorXd getVertArea(const Mesh& mesh);

void getWrinkledMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<std::complex<double>>& zvals, Eigen::MatrixXd& wrinkledV, double scaleRatio, bool isTangentCorrection);

void computeBaryGradient(const Eigen::Vector3d& P0, const Eigen::Vector3d& P1, const Eigen::Vector3d& P2, const Eigen::Vector3d& bary, Eigen::Matrix3d& baryGrad);

void mkdir(const std::string& foldername);
Eigen::VectorXd inconsistencyComputation(const Mesh& mesh, const Eigen::VectorXd& edgeW, const std::vector<std::complex<double>>& zval);
double getZListNorm(const std::vector<std::complex<double>>& zvals);

// convert to sorted edge omega, where we sort omega by edge's first vertex id
std::map<std::pair<int, int>, int> edgeMap(const std::vector< std::vector<int>>& edgeToVert);
std::map<std::pair<int, int>, int> edgeMap(const Eigen::MatrixXi& faces);

Eigen::VectorXd swapEdgeVec(const std::vector< std::vector<int>>& edgeToVert, const Eigen::VectorXd& edgeVec, int isInputConsistent = 1);
/* Swap the edgeVec to a "sorted" vec, where we sort the edgeVec based on the sorted edgToVert (based on the first edge vertex id), if isInputConsistent = 1.
 * If isInputConsistent = 0, we assume edgeVec is a "sorted" vec, and convert it back based on the order of edgeToVert vector.
*/
Eigen::VectorXd swapEdgeVec(const Eigen::MatrixXi& faces, const Eigen::VectorXd& edgeVec, int isInputConsistent = 1);


// save for render
void saveDphi4Render(const Eigen::MatrixXd& faceOmega, const Mesh& secMesh, const std::string& filename);
void saveAmp4Render(const Eigen::VectorXd& vertAmp, const std::string& filename, double ampMin = 0, double ampMax = 1);
void savePhi4Render(const Eigen::VectorXd& vertPhi, const std::string& fileName);
void saveFlag4Render(const Eigen::VectorXi& faceFlags, const std::string& filename); // 1 for selected, -1 for interface, 0 for unchanged
void saveSourcePts4Render(const Eigen::VectorXi& vertFlags, const Eigen::MatrixXd& vertVecs, const Eigen::VectorXd& vertAmp, const std::string& flagfilename);
