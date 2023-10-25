#pragma once
#include "BaseLoop.h"

namespace ComplexWrinkleField {
    class ComplexLoop : public BaseLoop	// We modify the Loop.h
    {
    public:
        void virtual BuildComplexS0(const VectorX& omega, ComplexSparseMatrixX& A) const override;

    private:
        std::vector<std::complex<double>> ComputeComplexWeight(const std::vector<Vector3>& pList, const std::vector<Vector3>& gradThetaList, const std::vector<double>& pWeights) const;
        std::vector<std::complex<double>> ComputeEdgeComplexWeight(const VectorX& omega, const Vector2& bary, int eid) const;
        std::vector<std::complex<double>> ComputeTriangleComplexWeight(const VectorX& omega, const Vector3& bary, int fid) const;

        std::vector<std::complex<double>> ComputeComplexWeight(const std::vector<double>& dthetaList, const std::vector<double>& coordList) const;
        Eigen::Vector3d ComputeBaryGradThetaFromOmegaPerface(const VectorX& omega, int fid, const Vector3& bary) const;
        Eigen::Vector3d ComputeGradThetaFromOmegaPerface(const VectorX& omega, int fid, int vInF) const;
    };
}

