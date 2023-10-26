#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "MeshLib/Mesh.h"

#ifndef GRAIN_SIZE
#define GRAIN_SIZE 10
#endif

Eigen::Vector3d ComputeHatWeight(double u, double v);

MatrixX SPDProjection(MatrixX A);

VectorX VertexVec2IntrinsicVec(const MatrixX& v, const Mesh& mesh);

VectorX FaceVec2IntrinsicEdgeVec(const MatrixX& v, const Mesh& mesh);
MatrixX IntrinsicEdgeVec2FaceVec(const VectorX& v, const Mesh& mesh);

VectorX GetFaceArea(const Mesh& mesh);
VectorX GetEdgeArea(const Mesh& mesh);
VectorX GetVertArea(const Mesh& mesh);

void GetWrinkledMesh(const MatrixX& V, const Eigen::MatrixXi& F, const VectorX& zvals, MatrixX& wrinkledV, double scaleRatio, bool isTangentCorrection);

void ComputeBaryGradient(const Vector3& P0, const Vector3& P1, const Vector3& P2, const Vector3& bary, Matrix3& baryGrad);

void Mkdir(const std::string& foldername);
VectorX InconsistencyComputation(const Mesh& mesh, const VectorX& edgeW, const VectorX& zval);
void RescaleZvals(const VectorX& normalizedZvals, const VectorX& norm, VectorX& zvals);
VectorX NormalizeZvals(const VectorX& zvals);

// Convert to sorted edge omega, where we sort omega by edge's first vertex id
std::map<std::pair<int, int>, int> EdgeMap(const std::vector<std::vector<int>>& edgeToVert);
std::map<std::pair<int, int>, int> EdgeMap(const Eigen::MatrixXi& faces);

VectorX SwapEdgeVec(const std::vector<std::vector<int>>& edgeToVert, const VectorX& edgeVec, int isInputConsistent = 1);
/* Swap the edgeVec to a "sorted" vec, where we sort the edgeVec based on the sorted edgeToVert (based on the first edge vertex id), if isInputConsistent = 1.
 * If isInputConsistent = 0, we assume edgeVec is a "sorted" vec, and convert it back based on the order of edgeToVert vector.
*/
VectorX SwapEdgeVec(const Eigen::MatrixXi& faces, const VectorX& edgeVec, int isInputConsistent = 1);

