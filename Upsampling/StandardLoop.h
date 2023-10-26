#pragma once

#include "BaseLoop.h"

namespace ComplexWrinkleField {
// The Standard Loop Rules
class StandardLoop : public BaseLoop {
public:
  // For the real value Loop, nrows = nVerts + nEdges, ncols = nVerts
  void virtual GetS0Size(int& nrows, int& ncols) const override {
    assert(_mesh && _mesh->IsTriangulated());
    nrows = _mesh->GetVertCount() + _mesh->GetEdgeCount();
    ncols = _mesh->GetVertCount();
  }
  

protected:
  virtual void _AssembleVertEvenInterior(int vi, TripletInserter out) const override;
  virtual void _AssembleVertEvenBoundary(int vi, TripletInserter out) const override;
  virtual void _AssembleVertOddInterior(int edge, TripletInserter out) const override;
  virtual void _AssembleVertOddBoundary(int edge, TripletInserter out) const override;
};
} // namespace ComplexWrinkleField
