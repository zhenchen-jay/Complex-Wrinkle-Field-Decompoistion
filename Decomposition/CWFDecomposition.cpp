#include "CWFDecomposition.h"
#include <igl/per_vertex_normals.h>
void CWFDecomposition::intialization(const Mesh &baseMesh, const std::vector<std::complex<double>> &unitZvals,
                                     const Eigen::VectorXd &amp, const Eigen::VectorXd &omega, int upsampleTimes)
{
    _baseMesh = baseMesh;
    _unitZvals = unitZvals;
    _amp = amp;
    _omega = omega;
    _upsampleTimes = upsampleTimes;
    _subOp = std::make_shared<ComplexLoop>();
    _subOp->SetMesh(_baseMesh);
    _upMesh = _subOp->meshSubdivide(upsampleTimes);
    _upMesh.GetPos(_upV);
    _upMesh.GetFace(_upF);
    igl::per_vertex_normals(_upV, _upF, _upN);
}

void CWFDecomposition::getCWF(Mesh &baseMesh, std::vector<std::complex<double>> &unitZvals, Eigen::VectorXd &amp,
                              Eigen::VectorXd &omega)
{
    baseMesh = _baseMesh;
    unitZvals = _unitZvals;
    amp = _amp;
    omega = _omega;
}

void CWFDecomposition::convertCWF2Variables(Eigen::VectorXd &x)
{
    int nverts = _baseMesh.GetVertCount();
    int nedges = _baseMesh.GetEdgeCount();

    x.setZero(3 * nverts + nedges);
    for(int i = 0; i < nverts; i++)
    {
        x[3 * i] = _amp[i];
        x[3 * i + 1] = _unitZvals[i].real();
        x[3 * i + 2] = _unitZvals[i].imag();
    }

    for(int i = 0; i < nedges; i++)
    {
        x[i + 3 * nverts] = _omega[i];
    }
}

void CWFDecomposition::convertVariables2CWF(const Eigen::VectorXd &x)
{
    int nverts = _baseMesh.GetVertCount();
    int nedges = _baseMesh.GetEdgeCount();

    if(_omega.rows() != nedges)
        _omega.setZero(nedges);
    if(_amp.rows() != nverts)
        _amp.setZero(nverts);
    if(_unitZvals.size() != nverts)
        _unitZvals.resize(nverts);

    for(int i = 0; i < nverts; i++)
    {
         _amp[i] = x[3 * i];
         _unitZvals[i] = std::complex<double>(x[3 * i + 1], x[3 * i + 2]);
    }

    for(int i = 0; i < nedges; i++)
    {
        _omega[i] = x[i + 3 * nverts];
    }
}

double CWFDecomposition::computeDifferenceEnergy(const Eigen::VectorXd &x, Eigen::VectorXd *grad,
                                                 Eigen::SparseMatrix<double> *hess)
{
    auto energyComputation = [&](const Eigen::VectorXd& x)
    {
        convertVariables2CWF(x);
        std::vector<std::complex<double>> zvals, upZvals;
        Eigen::VectorXd upOmega;

        _subOp->CWFSubdivide(_omega, zvals, upOmega, upZvals, _upsampleTimes);
        Eigen::MatrixXd wrinkledPos = _upV;
        double energy = 0;
        for(int i = 0; i < _upV.rows(); i++)
        {
            wrinkledPos.row(i) += upZvals[i].real() * _upN.row(i);
            energy += 0.5 * (wrinkledPos.row(i) - _wrinkledV.row(i)).squaredNorm();
        }
        return energy;
    };
    double energy = energyComputation(x);

    if(grad)
    {
        // finite difference
        int nvars = x.rows();
        grad->setZero(nvars);

        for(int i = 0; i < nvars; i++)
        {
            double eps = 1e-6;
            auto x1 = x, x2 = x;
            x1[i] = x[i] + eps;
            x2[i] = x[i] - eps;

            double energy1 = energyComputation(x1);
            double energy2 = energyComputation(x2);

            double val = (energy1 - energy2) / (2 * eps);   // finite difference
            (*grad)[i] = val;
        }

    }

    return energy;

}

void CWFDecomposition::optimizeCWF()
{
    Eigen::VectorXd x0;
    convertVariables2CWF(x0);

    for(int i = 0; i < 1000; i++)
    {
        // gradient descent

    }

}