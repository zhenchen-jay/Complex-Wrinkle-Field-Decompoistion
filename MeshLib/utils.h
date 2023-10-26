#pragma once

#include "types.h"

void ConvertToVector3( const MatrixX& X, std::vector<Vector3>& P);

void ConvertToMatrix(const std::vector<Vector3>& P, MatrixX& X);

// Return index of "query" in "values"
// Return -1 if "query" not found
int SearchIndex( const std::vector<int>& values, int query);
