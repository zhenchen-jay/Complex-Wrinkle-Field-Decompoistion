#pragma once

#include "../../CommonTools.h"
#include "../../ExtractIsoline.h"


void getBndProjectionMatrix(const MatrixX& pos, const Eigen::MatrixXi& faces, SparseMatrixX& bndProjM, SparseMatrixX* interiorProjM = nullptr, std::vector<int>* bndVids = nullptr, std::vector<bool> *bndFlags = nullptr);
void basemeshExtraction(const Mesh& wrinkledMesh, MatrixX& basemeshPos, Eigen::MatrixXi& basemeshFaces, MatrixX* isoPos = nullptr, Eigen::MatrixXi* isoEdges = nullptr);
