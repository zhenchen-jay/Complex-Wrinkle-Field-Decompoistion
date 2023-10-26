#pragma once

#include "MeshLib/Mesh.h"
#include "MeshLib/types.h"

namespace ComplexWrinkleField {
class CWF {
public:
  CWF() {}
  CWF(const VectorX& amp, const VectorX& omega, const VectorX& zvals, const Mesh& mesh) {
    Initialization(amp, omega, zvals, mesh);
  }

  void Initialization(const VectorX& amp, const VectorX& omega, const VectorX& zvals, const Mesh& mesh) {
    _amp = amp;
    _omega = omega;
    _zvals = zvals;
    _mesh = mesh;
  }

  VectorX _amp;   // wrinkle amplitude, stored as per vertex scalar
  VectorX _omega; // wrinkle frequency, stored as per edge one form
  VectorX _zvals; // wrinkle phase, stored as per vertex complex. _zvals = [x_0, x_1, ..., x_n, y_0, y_1, ..., y_n]
  Mesh _mesh;     // base mesh geometry
};
} // namespace ComplexWrinkleField
