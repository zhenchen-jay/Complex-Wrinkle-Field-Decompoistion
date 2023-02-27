#pragma once
#include "../../CommonTools.h"
#include "../../Upsampling/ComplexLoop.h"
#include "../../Upsampling/BaseLoop.h"
#include "../../TFWShell/TFWShell.h"

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
    void initialization(const CWF& cwf, int upsampleTimes);
    void initialization(const CWF& cwf, int upSampleTimes,
                        const Mesh& restMesh,               // rest (coarse) mesh
                        const Mesh& restWrinkleMesh,        // rest (wrinkle) mesh
                        const Mesh& wrinkledMesh,           // target wrinkle mesh (for decomposition)
                        double youngsModulus,               // Young's Modulus
                        double poissonRatio,                // Poisson's Ratio
                        double thickness                    // thickness
                        );

    void getCWF(CWF &baseCWF);

    void optimizeCWF();

    void optimizeAmpOmega();
    void optimizePhase();
    void optimizeBasemesh();

    double computeDifferenceEnergy(const VectorX& x, VectorX *grad = NULL, Eigen::SparseMatrix<double> *hess = NULL);

private:
    void convertCWF2Variables(VectorX& x);
    void convertVariables2CWF(const VectorX& x);

    void convertCWF2Variables(const CWF& cwf, VectorX& x);
    void convertVariables2CWF(const VectorX& x, CWF& cwf);


private:
    CWF _baseCWF;
    Mesh _upMesh, _restMesh, _restWrinkledMesh, _wrinkledMesh;
    Eigen::MatrixXd _upV, _upN, _wrinkledV;
    Eigen::MatrixXi _upF, _wrinkledF;

    int _upsampleTimes;

    std::shared_ptr<BaseLoop> _subOp;       // somehow we may need take the differential in the future (really nasty)
    TFWShell tfwShell;
    // material parameters
    double _youngsModulus;
    double _poissonRatio;
    double _thickness;
};