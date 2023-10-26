#pragma once
#include "BaseLoop.h"

namespace ComplexWrinkleField {
// The Complex Loop Rules given in [Chen et al, 2023]
class ComplexLoop : public BaseLoop {
public:
  // For the complex value (CWF) Loop, nrows = 2 * (nVerts + nEdges), ncols = 2 * nVerts
  void virtual GetS0Size(int& nrows, int& ncols) const override {
    assert(_mesh && _omega && _mesh->IsTriangulated() && _mesh->GetEdgeCount() == _omega->size());
    nrows = 2 * (_mesh->GetVertCount() + _mesh->GetEdgeCount());
    ncols = 2 * _mesh->GetVertCount();
  }

protected:
  virtual void _AssembleVertEvenInterior(int vi, TripletInserter out) const override;
  virtual void _AssembleVertEvenBoundary(int vi, TripletInserter out) const override;
  virtual void _AssembleVertOddInterior(int edge, TripletInserter out) const override;
  virtual void _AssembleVertOddBoundary(int edge, TripletInserter out) const override;

private:
  std::vector<std::complex<Scalar>> ComputeComplexWeightFromGradTheta(const std::vector<Vector3>& pList,
                                                                      const std::vector<Vector3>& gradThetaList,
                                                                      const std::vector<Scalar>& pWeights) const;
  std::vector<std::complex<Scalar>> ComputeEdgeComplexWeight(const Vector2& bary, int eid) const;
  std::vector<std::complex<Scalar>> ComputeTriangleComplexWeight(const Vector3& bary, int fid) const;

  std::vector<std::complex<Scalar>> ComputeComplexWeight(const std::vector<Scalar>& dthetaList,
                                                         const std::vector<Scalar>& coordList) const;
  Eigen::Vector3d ComputeBaryGradThetaFromOmegaPerface(int fid, const Vector3& bary) const;
  Eigen::Vector3d ComputeGradThetaFromOmegaPerface(int fid, int vInF) const;

  void inline FillTriplets(const int row, const int col, const int rowOffset, const int colOffset,
                           const std::complex<Scalar>& val, TripletInserter out) const {
    *out++ = TripletX(row, col, val.real());
    *out++ = TripletX(row + rowOffset, col + colOffset, val.real());
    *out++ = TripletX(row, col + colOffset, -val.imag());
    *out++ = TripletX(row + rowOffset, col, val.imag());
  }
};
} // namespace ComplexWrinkleField
