#pragma once

#include "CommonTools.h"
#include "MeshLib/Mesh.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

void ComputeEdgeMatrix(const Mesh& mesh, const VectorX& edgeW, const VectorX& edgeWeight, const int nverts,
                       SparseMatrixX& A);
void ComputeEdgeMatrixGivenMag(const Mesh& mesh, const VectorX& edgeW, const VectorX& vertAmp,
                               const VectorX& edgeWeight, const int nverts, SparseMatrixX& A);

void RoundZvalsFromEdgeOmega(const Mesh& mesh, const VectorX& edgeW, const VectorX& edgeWeight, const VectorX& vertArea,
                             int nverts, VectorX& zvals);
void RoundZvalsFromEdgeOmegaVertexMag(const Mesh& mesh, const VectorX& edgeW, const VectorX& vertAmp,
                                      const VectorX& edgeWeight, const VectorX& vertArea, int nverts, VectorX& zvals);
