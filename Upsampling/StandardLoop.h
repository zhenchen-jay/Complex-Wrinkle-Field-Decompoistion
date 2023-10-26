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
  // Loop rules for interior even vertices
  virtual void AssembleVertEvenInterior(int vi, TripletInserter out) const override;

  // Loop rules for boundary even vertices
  virtual void AssembleVertEvenBoundary(int vi, TripletInserter out) const override;

  // Loop rules for interior odd vertices
  virtual void AssembleVertOddInterior(int edge, TripletInserter out) const override;

  // Loop rules for boundary odd vertices
  virtual void AssembleVertOddBoundary(int edge, TripletInserter out) const override;
};
} // namespace ComplexWrinkleField
