#pragma once
#include "BaseLoop.h"

class ComplexLoop : public BaseLoop	// We modify the Loop.h
{
public:
    void virtual CWFSubdivide(
        const CWF& cwf,								// input CWF
        CWF& upcwf,									// output CWF
        int level,
        SparseMatrixX* upS0 = NULL,					// upsampled matrix for scalar
        SparseMatrixX* upS1 = NULL,					// upsampled matrix for one form
        ComplexSparseMatrixX* upComplexS0 = NULL	// upsampled matrix for complex scalar
    ) override;
    void virtual BuildComplexS0(const VectorX& omega, ComplexSparseMatrixX& A) override;
    
private:
    std::vector<std::complex<double>> computeComplexWeight(const std::vector<Vector3>& pList, const std::vector<Vector3>& gradThetaList, const std::vector<double>& pWeights);
    std::vector<std::complex<double>> computeEdgeComplexWeight(const VectorX& omega, const Vector2& bary, int eid);
    std::vector<std::complex<double>> computeTriangleComplexWeight(const VectorX& omega, const Vector3& bary, int fid);

    std::vector<std::complex<double>> computeComplexWeight(const std::vector<double>& dthetaList, const std::vector<double>& coordList);
    Eigen::Vector3d computeBaryGradThetaFromOmegaPerface(const VectorX& omega, int fid, const Vector3& bary);
    Eigen::Vector3d computeGradThetaFromOmegaPerface(const VectorX& omega, int fid, int vInF);
};
