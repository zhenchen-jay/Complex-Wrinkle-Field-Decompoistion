#pragma once
#include "../../CommonTools.h"
#include "../../Upsampling/ComplexLoop.h"
#include "../../Upsampling/BaseLoop.h"
#include "../../TFWShell/TFWShell.h"

#include <unordered_set>

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
                        bool isFixedBnd,                            // fixed bnd for loop
                        const Mesh& restMesh,                       // rest (coarse) mesh
                        const Mesh& baseMesh,                       // base (coarse) mesh
                        const Mesh& restWrinkleMesh,                // rest (wrinkle) mesh
                        const Mesh& wrinkledMesh,                   // target wrinkle mesh (for decomposition)
                        double youngsModulus,                       // Young's Modulus
                        double poissonRatio,                        // Poisson's Ratio
                        double thickness,                           // thickness
                        const std::unordered_set<int>& clampedVert  // clamped vertices
    );

    void initializeAmpOmega(const Eigen::MatrixXd& curPos, MeshConnectivity& curMeshCon, double ampGuess, Eigen::VectorXd& amp, Eigen::MatrixXd& faceOmega);
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
    void buildProjectionMat(const std::unordered_set<int>& clampedVerts, const MeshConnectivity& meshCon, int nverts);
    void updateWrinkleCompUpMat();


private:
    CWF _baseCWF;
    Mesh _upMesh, _restMesh, _restWrinkledMesh, _wrinkledMesh;
    MatrixX _upV, _upN, _wrinkledV;
    Eigen::MatrixXi _upF, _wrinkledF;

    ComplexSparseMatrixX _upZMat;
    SparseMatrixX _wrinkleCompUpMat, _LoopS0;
    
    std::unordered_set<int> _clampedVertices;
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

    // precomputations for amp omega
    std::unordered_set<int> _clampedAmpOmega;
    SparseMatrixX _projTFWMat;         // handle the fix verts
    SparseMatrixX _unprojTFWMat;
    int _nFreeAmp;
    int _nFreeOmega;

    // precomputations for phase
    SparseMatrixX _zvalDiffHess, _zvalCompHess;
    VectorX _zvalDiffCoeff;

    // precomputations for basemesh
    MatrixX _normalWrinkleUpdates;
    SparseMatrixX _baseMeshDiffHess, _projPosMat, _unprojPosMat;
    VectorX _baseMeshDiffCoeff;
};