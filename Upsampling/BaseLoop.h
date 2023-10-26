#pragma once
#include <iomanip>

#include "../MeshLib/Mesh.h"

namespace ComplexWrinkleField {
class BaseLoop {
public:
  BaseLoop() : _mesh(nullptr), _isFixBnd(false), _omega(nullptr) {}
  virtual ~BaseLoop() = default;
  inline void SetMesh(const Mesh& mesh) { _mesh = &mesh; }
  inline void SetOmega(const VectorX& omega) { _omega = &omega; }
  inline void SetBndFixFlag(const bool& isFixBnd) { _isFixBnd = isFixBnd; }

  // For the real value Loop, nrows = nVerts + nEdges, ncols = nVerts
  // For the complex value (CWF) Loop, nrows = 2 * (nVerts + nEdges), ncols = 2 * nVerts
  void virtual GetS0Size(int& nrows, int& ncols) const = 0;

  // Build the upsampling matrix for o-forms
  // 1. For real values: The Loop Scheme for 0-forms mentioned in "Subdivision exterior calculus for geometry processing", a.k.a [de Goes et al. 2016]
  // 2. For complex values: The Loop Scheme mentioned in "Complex Wrinkle Field Evolution", a.k.a. [Chen et al. 2023]
  // Remark: Loop subdivision for 0-form (per-vertex complex value):
  // Let the final upsampling matrix is U = V + iW, and the complex vector is z = x + i y,
  // x = [x_0, x_1, ..., x_n], y = [y_0, y_1, ..., y_n], then Uz = (Vx - Wy) + (Wx + Vy)i
  // or in other words, the upsampling scheme is:
  //      | V, -W |   | x |
  //      |       | * |   |
  //      | W,  V |   | y |
  // Therefore, our A is the matrix [V, -W, W, V]
  void BuildS0(SparseMatrixX& A) const;

  // Build the upsampling matrix for 1-forms
  // 1. The Loop Scheme for one form mentioned in "Subdivision exterior calculus for geometry processing", a.k.a [de Goes et al. 2016]
  // 2. For the fixed boundary case, the corresponding Loop Scheme is mentioned in the S.I. of "Complex Wrinkle Field Evolution", a.k.a. [Chen et al. 2023]
  void BuildS1(SparseMatrixX& A) const;

  // Get the edge information after subdivision
  void GetSubdividedEdges(std::vector<std::vector<int>>& edgeToVert) const;

  // Get the face information after subdivision
  void GetSubdividedFaces(std::vector<std::vector<int>>& faceToVert) const;

protected:
  Mesh const* _mesh;     // the mesh for loop
  bool _isFixBnd;        // whether we fix the boundary or not
  VectorX const* _omega; // the edge one forms

  // The vertex id map from old (input) mesh to the new upsampled mesh
  int GetVertVertIndex(int vert) const;

  // The new vertex id generated from the edge of the input
  int GetEdgeVertIndex(int edge) const;

  // The edge id map from the old (input) mesh to the new upsampled mesh
  int GetEdgeEdgeIndex(int edge, int vertInEdge) const;

  // The new edge id generated from the face of the input
  int GetFaceEdgeIndex(int face, int edgeInFace) const;

  // After upsampling, each face has been divided into four faces, return the new face id of the central face
  int GetCentralFaceIndex(int face) const;

  // After upsampling, each face has been divided into four faces, return the new face id of the vertex corner (given by vertInFace)
  int GetCornerFaceIndex(int face, int vertInFace) const;

  // Loop coefficient alpha, refer [de Goes et al. 2016]
  Scalar GetAlpha(int vert) const;

  // Loop coefficient beta, refer [de Goes et al. 2016]
  Scalar GetBeta(int vert) const;


  /************* Loop Rules for 0-forms *************/
  // Loop rules for interior even vertices
  virtual void AssembleVertEvenInterior(int vi, TripletInserter out) const = 0;

  // Loop rules for boundary even vertices
  virtual void AssembleVertEvenBoundary(int vi, TripletInserter out) const = 0;

  // Loop rules for interior odd vertices
  virtual void AssembleVertOddInterior(int edge, TripletInserter out) const = 0;

  // Loop rules for boundary odd vertices
  virtual void AssembleVertOddBoundary(int edge, TripletInserter out) const = 0;

  /************* Loop Rules for 1-forms *************/
  // Loop rules for interior even edges
  void AssembleEdgeEvenInterior(int edge, int vertInEdge, TripletInserter out) const;

  // Loop rules for boundary even edges (its endpoints are all on the boundary)
  void AssembleEdgeEvenBoundary(int edge, int vertInEdge, TripletInserter out) const;

  // Loop rules for partial boundary even edges (only one of its endpoints is on the boundary)
  void AssembleEdgeEvenPartialBoundary(int edge, int vertInEdge, TripletInserter out) const;

  // Loop rules for odd edges
  void AssembleEdgeOdd(int face, int edgeInFace, TripletInserter out) const;


  // Insert Triplet from edge-edge case for one-form subdivision
  void InsertEdgeEdgeValue(int row, int col, int vert, int rSign, Scalar val, TripletInserter out) const;
  // Insert Triplet from face-edge case for one-form subdivision
  void InsertEdgeFaceValue(int row, int face, int vert, int rSign, Scalar val, TripletInserter out) const;
};
} // namespace ComplexWrinkleField
