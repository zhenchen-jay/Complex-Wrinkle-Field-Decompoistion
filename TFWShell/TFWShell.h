#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <memory>

#include "../MeshLib/MeshConnectivity.h"
#include "../SecondFundamentalForm/SecondFundamentalFormDiscretization.h"
#include "../SecondFundamentalForm/MidedgeAverageFormulation.h"
#include "../SecondFundamentalForm/MidedgeAngleSinFormulation.h"
#include "../SecondFundamentalForm/MidedgeAngleTanFormulation.h"

#include "../MeshLib/GeometryDerivatives.h"
#include "../MeshLib/MeshGeometry.h"
#include "../CommonTools.h"

namespace  WrinkledTensionField {
    class TFWShell
    {
    public:
        TFWShell()
        {
            _restV.resize(0, 3);
            _baseV.resize(0, 3);
            _poissonRatio = 0;
            _thickness = 0;
            _youngsModulus = 0;
            _quadPts.resize(0);
        }
        TFWShell(const Eigen::MatrixXd &restV,
                 const MeshConnectivity &restMesh,
                 const Eigen::MatrixXd &baseV,
                 const MeshConnectivity &baseMesh,
                 double poissonRatio,
                 double thickness,
                 double youngsModulus,
                 int quadOrd = 1)
        {
            _restV = restV;
            _restMesh = restMesh;
            _baseV = baseV;
            _baseMesh = baseMesh;
            _poissonRatio = poissonRatio;
            _thickness = thickness;
            _youngsModulus = youngsModulus;
            _quadPts = buildQuadraturePoints(quadOrd);
        }

        void initialization();      // initialize all fundamental forms
        void updateBaseGeometries(const Eigen::MatrixXd &baseV);    // update base mesh positions and fundamental forms

        double stretchingEnergy(const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, Eigen::VectorXd *deriv, Eigen::SparseMatrix<double> *hessian, bool isProj = true);

        double bendingEnergy(const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, Eigen::VectorXd *deriv, Eigen::SparseMatrix<double> *hessian, bool isProj = true);

        double elasticReducedEnergy(const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, Eigen::VectorXd *deriv, Eigen::SparseMatrix<double> *hessian, bool isProj = true);

        double computeAmplitudesFromQuad(const Eigen::VectorXd& amp, int faceId, int quadId, Eigen::Vector2d *da, Eigen::Vector3d *gradA,
                                         Eigen::Matrix<double, 2, 3> *gradDA, Eigen::Matrix<double, 3, 3> *hessianA,
                                         std::vector<Eigen::Matrix<double, 3, 3>> *hessianDA);

        Eigen::Vector2d computeDphi(const Eigen::VectorXd& omega, int faceId, Eigen::Matrix<double, 2, 3> *gradDphi);

        Eigen::Matrix2d computeDaDphiTensor(const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, int faceId, int quadId, std::vector<Eigen::Matrix2d> *deriv,
                                            std::vector<Eigen::Matrix<double, 6, 6>> *hessian);

        Eigen::Matrix2d computeDphiDaTensor(const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, int faceId, int quadId, std::vector<Eigen::Matrix2d> *deriv,
                                            std::vector<Eigen::Matrix<double, 6, 6>> *hessian);

        Eigen::Matrix2d computeDaDaTensor(const Eigen::VectorXd& amp, int faceId, int quadId, std::vector<Eigen::Matrix2d> *deriv,
                                          std::vector<Eigen::Matrix<double, 3, 3>> *hessian);

        Eigen::Matrix2d computeDphiDphiTensor(const Eigen::VectorXd& omega, int faceId, int quadId, std::vector<Eigen::Matrix2d> *deriv,
                                              std::vector<Eigen::Matrix<double, 3, 3>> *hessian);


        std::vector<Eigen::Matrix2d>
        computeStretchingDensityFromQuad(const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, int faceId, int quadId, std::vector<Eigen::MatrixXd> *deriv,
                                         std::vector<std::vector<Eigen::MatrixXd> > *hessian);

        std::vector<Eigen::Matrix2d>
        computeBendingDensityFromQuad(const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, int faceId, int quadId, std::vector<Eigen::MatrixXd> *deriv,
                                      std::vector<std::vector<Eigen::MatrixXd> > *hessian);

        double stretchingEnergyPerface(const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, int faceId, Eigen::VectorXd *deriv, Eigen::MatrixXd *hessian, bool isProj = true);

        double bendingEnergyPerface(const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, int faceId, Eigen::VectorXd *deriv, Eigen::MatrixXd *hessian, bool isProj = true);

        Eigen::Matrix2d& getIbar(int faceId)
        {
            return _Ibars[faceId];
        }
        std::vector<Eigen::Matrix2d>& getIbars()
        {
            return _Ibars;
        }
        std::vector<Eigen::Matrix2d>& getIs()
        {
            return _Is;
        }

        Eigen::Matrix2d& getI(int faceId)
        {
            return _Is[faceId];
        }
        Eigen::Matrix2d& getIIbar(int faceId)
        {
            return _IIbars[faceId];
        }
        Eigen::Matrix2d& getII(int faceId)
        {
            return _IIs[faceId];
        }
        Eigen::MatrixXd& getMaximalCurvatureDir()
        {
            return _PD1;
        }
        Eigen::MatrixXd& getMinimalCurvatureDir()
        {
            return _PD2;
        }

    private:
        Eigen::MatrixXd _restV;             // rest position
        MeshConnectivity _restMesh;         // rest mesh
        Eigen::MatrixXd _baseV;             // current(base) position
        MeshConnectivity _baseMesh;         // current(base) mesh
        double _poissonRatio;               // Poisson's ratio
        double _thickness;                  // thickness
        double _youngsModulus;              // Young's modulus
        std::vector<QuadraturePoints> _quadPts;          // quadrature points

        // fundamental forms
        std::vector<Eigen::Matrix2d> _Ibars;
        std::vector<Eigen::Matrix2d> _IIbars;
        std::vector<Eigen::Matrix2d> _Is;
        std::vector<Eigen::Matrix2d> _IIs;

        // principal curvatures
        Eigen::VectorXd _PV1;       // tension
        Eigen::VectorXd _PV2;       // compresion
        Eigen::MatrixXd _PD1;
        Eigen::MatrixXd _PD2;
    };
}
