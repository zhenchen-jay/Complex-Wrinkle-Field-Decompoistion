#pragma once

#include "../../CommonTools.h"
#include "../../ExtractIsoline.h"

namespace ComplexWrinkleField {
    int getBndProjectionMatrix(const MatrixX& pos, const Eigen::MatrixXi& faces, SparseMatrixX& bndProjM, SparseMatrixX* interiorProjM = nullptr, std::vector<std::vector<int>>* bndVids = nullptr, std::vector<bool> *bndFlags = nullptr);
    void basemeshExtraction(const Mesh& wrinkledMesh, MatrixX& basemeshPos, Eigen::MatrixXi& basemeshFaces, MatrixX* isoPos = nullptr, Eigen::MatrixXi* isoEdges = nullptr);
    void basemeshHarmonicExtraction(const Mesh& wrinkledMesh, MatrixX& basemeshPos, Eigen::MatrixXi& basemeshFaces, MatrixX* isoPos = nullptr, Eigen::MatrixXi* isoEdges = nullptr);
}

