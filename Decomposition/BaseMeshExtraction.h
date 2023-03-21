#pragma once

#include "../../CommonTools.h"
#include "../../ExtractIsoline.h"

void basemeshExtraction(const Mesh& wrinkledMesh, MatrixX& basemeshPos, Eigen::MatrixXi& basemeshFaces, MatrixX* isoPos = nullptr, Eigen::MatrixXi* isoEdges = nullptr);
