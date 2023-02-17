#pragma once
#include "../../CommonTools.h"
#include "../../Upsampling/ComplexLoop.h"
#include "../../Upsampling/BaseLoop.h"

class CWFDecomposition
{
public:
    CWFDecomposition(){}
    CWFDecomposition(const Mesh& wrinkledMesh) 
    {
        setWrinkledMesh(wrinkledMesh);
    }

    void setWrinkledMesh(const Mesh& wrinkledMesh)
    {
        _wrinkledMesh = wrinkledMesh;
        _wrinkledMesh.GetPos(_wrinkledV);
        _wrinkledMesh.GetFace(_wrinkledF);
    }
    void intialization(const Mesh& baseMesh, const std::vector<std::complex<double>>& unitZvals, const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, int upsampleTimes);
    void getCWF(Mesh& baseMesh, std::vector<std::complex<double>>& unitZvals, Eigen::VectorXd& amp, Eigen::VectorXd& omega);

    void optimizeCWF();

    double computeDifferenceEnergy(const Eigen::VectorXd& x, Eigen::VectorXd *grad = NULL, Eigen::SparseMatrix<double> *hess = NULL);

private:
    void convertCWF2Variables(Eigen::VectorXd& x);
    void convertVariables2CWF(const Eigen::VectorXd& x);

    void convertCWF2Variables(const std::vector<std::complex<double>>& unitZvals, const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, Eigen::VectorXd& x);
    void convertVariables2CWF(const Eigen::VectorXd& x, std::vector<std::complex<double>>& unitZvals, Eigen::VectorXd& amp, Eigen::VectorXd& omega);


private:
    Mesh _baseMesh, _upMesh;
    Eigen::MatrixXd _upV, _upN, _wrinkledV;
    Eigen::MatrixXi _upF, _wrinkledF;

    std::vector<std::complex<double>> _unitZvals;
    Eigen::VectorXd _amp;
    Eigen::VectorXd _omega;
    int _upsampleTimes;

    std::shared_ptr<BaseLoop> _subOp;       // somehow we may need take the differential in the future (really nasty)
    Mesh _wrinkledMesh;
};