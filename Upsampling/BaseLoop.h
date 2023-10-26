#pragma once
#include "../CWFTypes.h"
#include "../MeshLib/Mesh.h"
#include <iomanip>

namespace ComplexWrinkleField {
class BaseLoop {
public:
  BaseLoop() : _mesh(nullptr), _isFixBnd(false), _omega(nullptr) {}
  virtual ~BaseLoop() = default;

  inline Mesh GetMesh() const { return *_mesh; }
  inline VectorX GetOmega() const { return *_omega; }
  inline void SetMesh(const Mesh& mesh) { _mesh = &mesh; }
  inline void SetOmega(const VectorX& omega) { _omega = &omega; }
  inline void SetBndFixFlag(const bool& isFixBnd) { _isFixBnd = isFixBnd; }

  void virtual GetS0Size(int& nrows, int& ncols) const = 0;
  // For the real value Loop, nrows = nVerts + nEdges, ncols = nVerts
  // For the complex value (CWF) Loop, nrows = 2 * (nVerts + nEdges), ncols = 2 * nVerts

  void BuildS0(SparseMatrixX& A) const;
  // 1. For real values: The Loop Scheme for 0-form mentioned in "Subdivision exterior calculus for geometry processing", a.k.a [de Goes et al. 2016]
  // 2. For complex values: The Loop Scheme mentioned in "Complex Wrinkle Field Evolution", a.k.a. [Chen et al. 2023]
  // Remark: Loop subdivision for 0-form (per-vertex complex value):
  // Let the final upsampling matrix is U = V + iW, and the complex vector is z = x + i y,
  // x = [x_0, x_1, ..., x_n], y = [y_0, y_1, ..., y_n], then Uz = (Vx - Wy) + (Wx + Vy)i
  // or in other words, the upsampling scheme is:
  //      | V, -W |   | x |
  //      |       | * |   |
  //      | W,  V |   | y |
  // Therefore, our A is the matrix [V, -W, W, V]
  // 

  void BuildS1(SparseMatrixX& A) const;
  // The Loop Scheme for one form mentioned in "Subdivision exterior calculus for geometry processing", a.k.a [de Goes et al. 2016]
  // For the fixed boundary case, the corresponding Loop Scheme is mentioned in the S.I. of "Complex Wrinkle Field Evolution", a.k.a. [Chen et al. 2023]

  void GetSubdividedEdges(std::vector<std::vector<int>>& edgeToVert) const;
  void GetSubdividedFaces(std::vector<std::vector<int>>& faceToVert) const;

  bool IsVertRegular(int vert) const;
  bool AreIrregularVertsIsolated() const;

protected:
  Mesh const* _mesh;     // the mesh for loop
  bool _isFixBnd;        // whether we fix the boundary or not
  VectorX const* _omega; // the edge one forms

  int _GetVertVertIndex(int vert) const;
  int _GetEdgeVertIndex(int edge) const;

  int _GetEdgeEdgeIndex(int edge, int vertInEdge) const;
  int _GetFaceEdgeIndex(int face, int edgeInFace) const;

  int _GetCentralFaceIndex(int face) const;
  int _GetCornerFaceIndex(int face, int vertInFace) const;

  Scalar _GetAlpha(int vert) const;
  Scalar _GetBeta(int vert) const;

  virtual void _AssembleVertEvenInterior(int vi, TripletInserter out) const = 0;
  virtual void _AssembleVertEvenBoundary(int vi, TripletInserter out) const = 0;
  virtual void _AssembleVertOddInterior(int edge, TripletInserter out) const = 0;
  virtual void _AssembleVertOddBoundary(int edge, TripletInserter out) const = 0;

  void _AssembleEdgeEvenInterior(int edge, int vertInEdge, TripletInserter out) const;
  void _AssembleEdgeEvenBoundary(int edge, int vertInEdge, TripletInserter out) const;
  void _AssembleEdgeEvenPartialBoundary(int edge, int vertInEdge, TripletInserter out) const;
  void _AssembleEdgeOdd(int face, int edgeInFace, TripletInserter out) const;

  void _InsertEdgeEdgeValue(int row, int col, int vert, int rSign, Scalar val, TripletInserter out) const;
  void _InsertEdgeFaceValue(int row, int face, int vert, int rSign, Scalar val, TripletInserter out) const;
};
} // namespace ComplexWrinkleField
