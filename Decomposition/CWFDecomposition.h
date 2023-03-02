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

    void initialization(int upSampleTimes,
                        bool isFixedBnd,                    // fixed bnd for loop
                        const Mesh& restMesh,               // rest (coarse) mesh
                        const Mesh& baseMesh,               // base (coarse) mesh
                        const Mesh& restWrinkleMesh,        // rest (wrinkle) mesh
                        const Mesh& wrinkledMesh,           // target wrinkle mesh (for decomposition)
                        double youngsModulus,               // Young's Modulus
                        double poissonRatio,                // Poisson's Ratio
                        double thickness                    // thickness
    );

    void initializeAmpOmega(const Eigen::MatrixXd& curPos, MeshConnectivity& curMeshCon, double ampGuess, Eigen::VectorXd& amp, Eigen::VectorXd& omega);
    /*
     * initialize amp and omega based on the compression amount.
     * REQUIRE: tfwshell has been initialized! (providing foundamental forms)
     */

    void getCWF(CWF &baseCWF);

    void optimizeCWF();

    void optimizeAmpOmega();
    void precomputationForPhase();
    void optimizePhase();
    void precomptationForBaseMesh();
    void optimizeBasemesh();


    // vertex phase update energies
    double computeDifferenceFromZvals(const ComplexVectorX& zvals, VectorX *grad = nullptr, SparseMatrixX *hess = nullptr);
    double computeCompatibilityEnergy(const VectorX& omega, const ComplexVectorX& zvals, VectorX* grad = nullptr, SparseMatrixX* hess = nullptr);
    // compute compatibility between omega (_baseCWF.omega) and zvals
    double computeUnitNormEnergy(const ComplexVectorX& zval, VectorX* grad = nullptr, SparseMatrixX* hess = nullptr, bool isProj = true);

    void testDifferenceFromZvals(const ComplexVectorX& zvals);

    // base mesh update energies
    double computeDifferenceFromBasemesh(const MatrixX& pos, VectorX* grad = nullptr, SparseMatrixX* hess = nullptr);
    VectorX flatMatrix(const MatrixX& x);
    MatrixX unFlatVector(const VectorX& v, Eigen::Index cols);
    void testDifferenceFromBasemesh(const MatrixX& pos);

private:
    void convertCWF2Variables(VectorX& x);
    void convertVariables2CWF(const VectorX& x);

    void convertCWF2Variables(const CWF& cwf, VectorX& x);
    void convertVariables2CWF(const VectorX& x, CWF& cwf);

    void updateWrinkleCompUpMat();


private:
    CWF _baseCWF;
    Mesh _upMesh, _restMesh, _restWrinkledMesh, _wrinkledMesh;
    MatrixX _upV, _upN, _wrinkledV;
    Eigen::MatrixXi _upF, _wrinkledF;

    ComplexSparseMatrixX _upZMat;
    SparseMatrixX _wrinkleCompUpMat, _LoopS0;

    int _upsampleTimes;

    std::shared_ptr<BaseLoop> _subOp;       // somehow we may need take the differential in the future (really nasty)
    TFWShell tfwShell;
    // material parameters
    double _youngsModulus;
    double _poissonRatio;
    double _thickness;
    // base mesh get edge (fixed as the initial ones)
    VectorX _baseEdgeArea, _baseVertArea;
    VectorX _upVertArea;

    // upsample Amp
    VectorX _upAmp;

    // precomputations for phase
    SparseMatrixX _zvalDiffHess, _zvalCompHess;
    VectorX _zvalDiffCoeff;

    // precomputations for basemesh
    MatrixX _normalWrinkleUpdates;
    SparseMatrixX _baseMeshDiffHess;
    VectorX _baseMeshDiffCoeff;
};