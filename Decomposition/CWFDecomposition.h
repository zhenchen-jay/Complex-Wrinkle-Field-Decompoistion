#pragma once
#include <unordered_set>

#include "../../CommonTools.h"
#include "../../Upsampling/ComplexLoop.h"
#include "../../Upsampling/BaseLoop.h"
#include "../../TFWShell/TFWShell.h"
#include "../../CWF.h"

namespace ComplexWrinkleField {
    class CWFDecomposition {
    public:
        CWFDecomposition() {}
        CWFDecomposition(const Mesh &wrinkledMesh) {
            SetWrinkledMesh(wrinkledMesh);
        }

        void SetWrinkledMesh(const Mesh &wrinkledMesh) {
            _wrinkledMesh = wrinkledMesh;
            _wrinkledMesh.GetPos(_wrinkledV);
            _wrinkledMesh.GetFace(_wrinkledF);
        }
        void Initialization(const CWF &cwf, int upsampleTimes);
        void Initialization(const CWF &cwf, int upSampleTimes,
                            const Mesh &restMesh,               // rest (coarse) mesh
                            const Mesh &restWrinkleMesh,        // rest (wrinkle) mesh
                            const Mesh &wrinkledMesh,           // target wrinkle mesh (for decomposition)
                            double youngsModulus,               // Young's Modulus
                            double poissonRatio,                // Poisson's Ratio
                            double thickness                    // thickness
        );

        void Initialization(int upSampleTimes,
                            bool isFixedBnd,                            // fixed bnd for loop
                            const Mesh &restMesh,                       // rest (coarse) mesh
                            const Mesh &baseMesh,                       // base (coarse) mesh
                            const Mesh &restWrinkleMesh,                // rest (wrinkle) mesh
                            const Mesh &wrinkledMesh,                   // target wrinkle mesh (for decomposition)
                            double youngsModulus,                       // Young's Modulus
                            double poissonRatio,                        // Poisson's Ratio
                            double thickness,                           // thickness
                            const std::unordered_set<int> &clampedVert  // clamped vertices
        );

        void Initialization(const CWF &cwf, int upSampleTimes,
                            bool isFixedBnd,                            // fixed bnd for loop
                            const Mesh &restMesh,                       // rest (coarse) mesh
                            const Mesh &restWrinkleMesh,                // rest (wrinkle) mesh
                            const Mesh &wrinkledMesh,                   // target wrinkle mesh (for decomposition)
                            double youngsModulus,                       // Young's Modulus
                            double poissonRatio,                        // Poisson's Ratio
                            double thickness,                           // thickness
                            const std::unordered_set<int> &clampedVert  // clamped vertices
        );

        void InitializeAmpOmega(const Eigen::MatrixXd &curPos, MeshConnectivity &curMeshCon, double ampGuess, Eigen::VectorXd &amp, Eigen::MatrixXd &faceOmega);
        /*
         * Initialize amp and omega based on the compression amount.
         * REQUIRE: tfwshell has been initialized! (providing fundamental forms)
         */

        void GetCWF(CWF &baseCWF);

        void OptimizeCWF();

        void OptimizeAmpOmega();
        void PrecomputationForPhase();
        void OptimizePhase();
        void PrecomputationForBaseMesh();
        void OptimizeBasemesh();

        // Vertex phase update energies
        double ComputeDifferenceFromZvals(const VectorX &zvals, VectorX *grad = nullptr, SparseMatrixX *hess = nullptr);
        double ComputeCompatibilityEnergy(const VectorX &omega, const VectorX &zvals, VectorX *grad = nullptr, SparseMatrixX *hess = nullptr);
        // Compute compatibility between omega (_baseCWF.omega) and zvals
        double ComputeUnitNormEnergy(const VectorX &zval, VectorX *grad = nullptr, SparseMatrixX *hess = nullptr, bool isProj = true);

        void TestDifferenceFromZvals(const VectorX &zvals);

        // Base mesh update energies
        double ComputeDifferenceFromBasemesh(const MatrixX &pos, VectorX *grad = nullptr, SparseMatrixX *hess = nullptr);
        VectorX FlatMatrix(const MatrixX &x);
        MatrixX UnFlatVector(const VectorX &v, Eigen::Index cols);
        void TestDifferenceFromBasemesh(const MatrixX &pos);

    private:
        void BuildProjectionMat(const std::unordered_set<int> &clampedVerts, const MeshConnectivity &meshCon, int nverts);
        void UpdateWrinkleCompUpMat();

    private:
        CWF _baseCWF;
        Mesh _upMesh, _restMesh, _restWrinkledMesh, _wrinkledMesh;
        MatrixX _upV, _upN, _wrinkledV;
        Eigen::MatrixXi _upF, _wrinkledF;

        SparseMatrixX _upZMat;
        SparseMatrixX _wrinkleCompUpMat, _LoopS0;

        std::unordered_set<int> _clampedVertices;
        int _upsampleTimes;

        WrinkledTensionField::TFWShell _tfwShell;
        // Material parameters
        double _youngsModulus;
        double _poissonRatio;
        double _thickness;
        // Base mesh get edge (fixed as the initial ones)
        VectorX _baseEdgeArea, _baseVertArea;
        VectorX _upVertArea;

        // Upsample Amp
        VectorX _upAmp;

        // Precomputations for amp omega
        std::unordered_set<int> _clampedAmpOmega;
        SparseMatrixX _projTFWMat;         // Handle the fix verts
        SparseMatrixX _unprojTFWMat;
        int _nFreeAmp;
        int _nFreeOmega;

        // Precomputations for phase
        SparseMatrixX _zvalDiffHess, _zvalCompHess;
        VectorX _zvalDiffCoeff;

        // Precomputations for base mesh
        MatrixX _normalWrinkleUpdates;
        SparseMatrixX _baseMeshDiffHess, _projPosMat, _unprojPosMat;
        VectorX _baseMeshDiffCoeff;
    };
}
