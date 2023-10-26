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
  // Loop rules for interior even vertices
  virtual void AssembleVertEvenInterior(int vi, TripletInserter out) const override;

  // Loop rules for boundary even vertices
  virtual void AssembleVertEvenBoundary(int vi, TripletInserter out) const override;

  // Loop rules for interior odd vertices
  virtual void AssembleVertOddInterior(int edge, TripletInserter out) const override;

  // Loop rules for boundary odd vertices
  virtual void AssembleVertOddBoundary(int edge, TripletInserter out) const override;

private:
  // Barycentrically blend the z values from the corner of the polygon:
  // the angle contribution from each point given by dthetaList[i]. Then z = \sum pWeight[i] * exp(dthetaList[i]) * z(p_i).
  // This is the generalization of Equation (19) in [Chen et al, 2023]
  // Return a vector of weights: pWeight[i] * exp(dthetaList[i]), i = 0, 1, 2, ...
  std::vector<std::complex<Scalar>> ComputeComplexWeight(const std::vector<Scalar>& dthetaList,
                                                         const std::vector<Scalar>& coordList) const;

  // Barycentrically blend the z values from the corner of the polygon:
  // The new position is given as \sum pWeights[i] * pList[i], the angle contribution for the i-th point is:
  // dtheta_i = gradThetaList[i].dot(p - pList[i]). Then z = \sum pWeight[i] * exp(dtheta_i) * z(p_i).
  // Return a vector of weights: pWeight[i] * exp(dtheta_i), i = 0, 1, 2, ...
  std::vector<std::complex<Scalar>> ComputeComplexWeightFromGradTheta(const std::vector<Vector3>& pList,
                                                                      const std::vector<Vector3>& gradThetaList,
                                                                      const std::vector<Scalar>& pWeights) const;

  // Barycentrially blender the z value, where the new position point is on the edge, Equation (18) in [Chen et al, 2023]
  std::vector<std::complex<Scalar>> ComputeEdgeComplexWeight(const Vector2& bary, int eid) const;

  // Barycentrially blender the z value, where the new position point is on the face, Equation (19) in [Chen et al, 2023]
  std::vector<std::complex<Scalar>> ComputeTriangleComplexWeight(const Vector3& bary, int fid) const;

  // For each vertex on the face, compute the approximate gradient of theta from edge 1-forms. Equation (52) in S.I.of [Chen et al, 2023]
  Eigen::Vector3d ComputeGradThetaFromOmegaPerfaceCorner(int fid, int vInF) const;

  // Barycentrically blend the gradient of theta from face corners
  Eigen::Vector3d ComputeBaryGradThetaFromOmegaPerface(int fid, const Vector3& bary) const;

  // Filling the triplets, where we use the fact that: U = V + iW, and the complex vector is z = x + i y,
  // x = [x_0, x_1, ..., x_n], y = [y_0, y_1, ..., y_n], then Uz = (Vx - Wy) + (Wx + Vy)i,
  // rowOffset = #new_vertices, colOffset = #old_vertices
  void inline FillTriplets(const int row, const int col, const int rowOffset, const int colOffset,
                           const std::complex<Scalar>& val, TripletInserter out) const {
    *out++ = TripletX(row, col, val.real());
    *out++ = TripletX(row + rowOffset, col + colOffset, val.real());
    *out++ = TripletX(row, col + colOffset, -val.imag());
    *out++ = TripletX(row + rowOffset, col, val.imag());
  }
};
} // namespace ComplexWrinkleField
