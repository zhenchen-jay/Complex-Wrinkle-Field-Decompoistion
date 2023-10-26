#pragma once

#include "CommonTools.h"
#include "MeshLib/Mesh.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

void computeEdgeMatrix(const Mesh& mesh, const VectorX& edgeW, const VectorX& edgeWeight, const int nverts,
                       SparseMatrixX& A);
void computeEdgeMatrixGivenMag(const Mesh& mesh, const VectorX& edgeW, const VectorX& vertAmp,
                               const VectorX& edgeWeight, const int nverts, SparseMatrixX& A);

void roundZvalsFromEdgeOmega(const Mesh& mesh, const VectorX& edgeW, const VectorX& edgeWeight, const VectorX& vertArea,
                             int nverts, VectorX& zvals);
void roundZvalsFromEdgeOmegaVertexMag(const Mesh& mesh, const VectorX& edgeW, const VectorX& vertAmp,
                                      const VectorX& edgeWeight, const VectorX& vertArea, int nverts, VectorX& zvals);
