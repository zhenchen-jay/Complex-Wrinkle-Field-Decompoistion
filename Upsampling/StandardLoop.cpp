#include "StandardLoop.h"

namespace ComplexWrinkleField {
/***** Standard Loop subdivision for 0-form (per-vertex real value) *****/
// Loop rules for interior even vertices
void StandardLoop::AssembleVertEvenInterior(int vi, TripletInserter out) const {
  // Fig1 left in S.I. of [de Goes et al. 2016]
  Scalar alpha = GetAlpha(vi);
  int row = GetVertVertIndex(vi);
  const std::vector<int>& edges = _mesh->GetVertEdges(vi);
  for (int k = 0; k < edges.size(); ++k) {
    int edge = edges[k];
    int viInEdge = _mesh->GetVertIndexInEdge(edge, vi);
    int vj = _mesh->GetEdgeVerts(edge)[(viInEdge + 1) % 2];
    *out++ = TripletX(row, vj, alpha);
  }
  *out++ = TripletX(row, vi, 1. - alpha * edges.size());
}

// Loop rules for boundary even vertices
void StandardLoop::AssembleVertEvenBoundary(int vi, TripletInserter out) const {
  // Fig1 right-top in S.I. of [de Goes et al. 2016]
  std::vector<int> boundary(2);
  boundary[0] = _mesh->GetVertEdges(vi).front();
  boundary[1] = _mesh->GetVertEdges(vi).back();

  int row = GetVertVertIndex(vi);

  if (_isFixBnd)
    *out++ = TripletX(row, vi, 1.0);
  else {
    for (int j = 0; j < boundary.size(); ++j) {
      int edge = boundary[j];
      assert(_mesh->IsEdgeBoundary(edge));
      int viInEdge = _mesh->GetVertIndexInEdge(edge, vi);
      int vj = _mesh->GetEdgeVerts(edge)[(viInEdge + 1) % 2];
      *out++ = TripletX(row, vj, 0.125);
    }
    *out++ = TripletX(row, vi, 0.75);
  }
}

// Loop rules for interior odd vertices
void StandardLoop::AssembleVertOddInterior(int edge, TripletInserter out) const {
  // Fig1 mid in S.I. of [de Goes et al. 2016]
  for (int j = 0; j < 2; ++j) {
    int face = _mesh->GetEdgeFaces(edge)[j];
    int offset = _mesh->GetEdgeIndexInFace(face, edge);

    int vi = _mesh->GetFaceVerts(face)[(offset + 0) % 3];
    int vj = _mesh->GetFaceVerts(face)[(offset + 1) % 3];
    int vk = _mesh->GetFaceVerts(face)[(offset + 2) % 3];

    int row = GetEdgeVertIndex(edge);
    *out++ = TripletX(row, vi, 0.1875);
    *out++ = TripletX(row, vj, 0.1875);
    *out++ = TripletX(row, vk, 0.125);
  }
}

// Loop rules for boundary odd vertices
void StandardLoop::AssembleVertOddBoundary(int edge, TripletInserter out) const {
  // Fig1 right-bot in S.I. of [de Goes et al. 2016]
  int row = GetEdgeVertIndex(edge);
  for (int j = 0; j < 2; ++j) {
    int vj = _mesh->GetEdgeVerts(edge)[j];
    *out++ = TripletX(row, vj, 0.5);
  }
}

} // namespace ComplexWrinkleField
