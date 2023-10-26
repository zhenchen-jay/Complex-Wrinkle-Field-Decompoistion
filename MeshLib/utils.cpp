#include "utils.h"

#include <cassert>
#include <iostream>

void
ConvertToVector3(
    const MatrixX& X,
    std::vector<Vector3>& P)
{
    P.resize(X.rows());
    #pragma omp parallel for
    for (int i = 0; i < (int) P.size(); ++i) 
    {
        P[i] = X.row(i).transpose();
    }
}

void
ConvertToMatrix(
    const std::vector<Vector3>& P,
    MatrixX& X)
{
    X = MatrixX::Zero(P.size(), 3);
    #pragma omp parallel for
    for (int i = 0; i < (int) P.size(); ++i) 
    {
        X.row(i) = P[i].transpose();
    }
}

int
SearchIndex(const std::vector<int>& values, int query)
{
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (values[i] == query) return i;
    }
    return -1;
}
