#pragma once
#include "BaseLoop.h"

class ComplexLoop : public BaseLoop	// We modify the Loop.h
{
public:
    void virtual CWFSubdivide(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, int level) override;
    void virtual BuildComplexS0(const Eigen::VectorXd& omega, Eigen::SparseMatrix<std::complex<double>>& A) override;
    
private:
    std::vector<std::complex<double>> computeComplexWeight(const std::vector<Eigen::Vector3d>& pList, const std::vector<Eigen::Vector3d>& gradThetaList, const std::vector<double>& pWeights);
    std::vector<std::complex<double>> computeEdgeComplexWeight(const Eigen::VectorXd& omega, const Eigen::Vector2d& bary, int eid);
    std::vector<std::complex<double>> computeTriangleComplexWeight(const Eigen::VectorXd& omega, const Eigen::Vector3d& bary, int fid);

    std::vector<std::complex<double>> computeComplexWeight(const std::vector<double>& dthetaList, const std::vector<double>& coordList);
    Eigen::Vector3d computeBaryGradThetaFromOmegaPerface(const Eigen::VectorXd& omega, int fid, const Eigen::Vector3d& bary);
    Eigen::Vector3d computeGradThetaFromOmegaPerface(const Eigen::VectorXd& omega, int fid, int vInF);
};
