#include "BaseLoop.h"
#include <cassert>
#include <memory>

namespace ComplexWrinkleField {
// The vertex id map from old (input) mesh to the new upsampled mesh
int BaseLoop::GetVertVertIndex(int vert) const { return vert; }

// The new vertex id generated from the edge of the input
int BaseLoop::GetEdgeVertIndex(int edge) const { return _mesh->GetVertCount() + edge; }

// The edge id map from the old (input) mesh to the new upsampled mesh
int BaseLoop::GetEdgeEdgeIndex(int edge, int vertInEdge) const { return 2 * edge + vertInEdge; }

// The new edge id generated from the face of the input
int BaseLoop::GetFaceEdgeIndex(int face, int edgeInFace) const {

  assert(_mesh->IsTriangulated());
  return 2 * _mesh->GetEdgeCount() + 3 * face + edgeInFace;
}

// After upsampling, each face has been divided into four faces, return the new face id of the central face
int BaseLoop::GetCentralFaceIndex(int face) const {

  assert(_mesh->IsTriangulated());
  return 4 * face + 3;
}

// After upsampling, each face has been divided into four faces, return the new face id of the vertex corner (given by vertInFace)
int BaseLoop::GetCornerFaceIndex(int face, int vertInFace) const {

  assert(_mesh->IsTriangulated());
  return 4 * face + vertInFace;
}

// Get the edge information after subdivision
void BaseLoop::GetSubdividedEdges(std::vector<std::vector<int>>& edgeToVert) const {

  assert(_mesh->IsTriangulated());

  int nEdges = _mesh->GetEdgeCount();
  int nFaces = _mesh->GetFaceCount();
  edgeToVert.resize(2 * nEdges + 3 * nFaces);

  for (int edge = 0; edge < nEdges; ++edge) {
    const std::vector<int>& eVerts = _mesh->GetEdgeVerts(edge);
    for (int i = 0; i < 2; ++i) {
      int v0 = GetVertVertIndex(eVerts[i]);
      int v1 = GetEdgeVertIndex(edge);
      if (v0 > v1) std::swap(v0, v1);

      int index = GetEdgeEdgeIndex(edge, i);
      edgeToVert[index].push_back(v0);
      edgeToVert[index].push_back(v1);
    }
  }

  for (int face = 0; face < nFaces; ++face) {
    const std::vector<int>& fEdges = _mesh->GetFaceEdges(face);
    for (int i = 0; i < 3; ++i) {
      int v0 = GetEdgeVertIndex(fEdges[i]);
      int v1 = GetEdgeVertIndex(fEdges[(i + 1) % 3]);
      if (v0 > v1) std::swap(v0, v1);

      int index = GetFaceEdgeIndex(face, i);
      edgeToVert[index].push_back(v0);
      edgeToVert[index].push_back(v1);
    }
  }
}

// Get the face information after subdivision
void BaseLoop::GetSubdividedFaces(std::vector<std::vector<int>>& faceToVert) const {

  assert(_mesh->IsTriangulated());

  int nFaces = _mesh->GetFaceCount();
  faceToVert.resize(4 * nFaces);

  std::vector<int> faceFlagsNew;
  faceFlagsNew.resize(4 * nFaces, 0);

  for (int face = 0; face < _mesh->GetFaceCount(); ++face) {
    int central = GetCentralFaceIndex(face);
    const std::vector<int>& fVerts = _mesh->GetFaceVerts(face);
    const std::vector<int>& fEdges = _mesh->GetFaceEdges(face);
    for (int j = 0; j < 3; ++j) {
      // Corner face
      int index = GetCornerFaceIndex(face, j);
      faceToVert[index].push_back(GetVertVertIndex(fVerts[j]));
      faceToVert[index].push_back(GetEdgeVertIndex(fEdges[j]));
      faceToVert[index].push_back(GetEdgeVertIndex(fEdges[(j + 2) % 3]));
      // Central face
      faceToVert[central].push_back(GetEdgeVertIndex(fEdges[j]));
    }
  }
}

// Loop coefficient alpha, refer [de Goes et al. 2016]
Scalar BaseLoop::GetAlpha(int vert) const {

  assert(!_mesh->IsVertBoundary(vert));
  const std::vector<int>& vEdges = _mesh->GetVertEdges(vert);

  // Fig1 in S.I. left [de Goes et al. 2016]
  Scalar alpha = 0.375;
  if (vEdges.size() == 3)
    alpha /= 2;
  else
    alpha /= vEdges.size();
  return alpha;
}

// Loop coefficient beta, refer [de Goes et al. 2016]
Scalar BaseLoop::GetBeta(int vert) const {

  assert(!_mesh->IsVertBoundary(vert));
  const std::vector<int>& vFaces = _mesh->GetVertFaces(vert);

  Scalar beta = 0.;
  if (vFaces.size() >= 6)
    beta = 0.25;
  else if (vFaces.size() == 5)
    beta = 0.25 - 0.0625 * std::pow(std::sin(0.4 * M_PI), 2);
  else if (vFaces.size() == 4)
    beta = 0.125;
  else if (vFaces.size() == 3)
    beta = 0.25 / 3.;
  else
    assert(false);
  return beta;
}

// The Loop Scheme for 0-forms
void BaseLoop::BuildS0(SparseMatrixX& A) const {
  assert(_mesh && _mesh->IsTriangulated());

  int nrows, ncols;
  GetS0Size(nrows, ncols);

  std::vector<TripletX> triplets;
  const int nverts = _mesh->GetVertCount();
  const int nEdges = _mesh->GetEdgeCount();

  // Even (old) verts
  for (int vi = 0; vi < nverts; ++vi) {
    if (_mesh->IsVertBoundary(vi))
      AssembleVertEvenBoundary(vi, std::back_inserter(triplets));
    else
      AssembleVertEvenInterior(vi, std::back_inserter(triplets));
  }

  // Odd (new) verts
  for (int edge = 0; edge < nEdges; ++edge) {
    if (_mesh->IsEdgeBoundary(edge))
      AssembleVertOddBoundary(edge, std::back_inserter(triplets));
    else
      AssembleVertOddInterior(edge, std::back_inserter(triplets));
  }

  A.resize(nrows, ncols);
  A.setFromTriplets(triplets.begin(), triplets.end());
}

// The Loop Scheme for 1-forms
void BaseLoop::BuildS1(SparseMatrixX& A) const {

  assert(_mesh->IsTriangulated());

  std::vector<TripletX> triplets;
  int nEdges = _mesh->GetEdgeCount();
  int nFaces = _mesh->GetFaceCount();

  for (int edge = 0; edge < nEdges; ++edge) {
    if (_mesh->IsEdgeBoundary(edge)) {
      AssembleEdgeEvenBoundary(edge, 0, std::back_inserter(triplets));
      AssembleEdgeEvenBoundary(edge, 1, std::back_inserter(triplets));
    } else {
      const std::vector<int>& eVerts = _mesh->GetEdgeVerts(edge);
      for (int i = 0; i < eVerts.size(); ++i) {
        if (_mesh->IsVertBoundary(eVerts[i]))
          AssembleEdgeEvenPartialBoundary(edge, i, std::back_inserter(triplets));
        else
          AssembleEdgeEvenInterior(edge, i, std::back_inserter(triplets));
      }
    }
  }

  for (int face = 0; face < nFaces; ++face) {
    AssembleEdgeOdd(face, 0, std::back_inserter(triplets));
    AssembleEdgeOdd(face, 1, std::back_inserter(triplets));
    AssembleEdgeOdd(face, 2, std::back_inserter(triplets));
  }

  A.resize(2 * nEdges + 3 * nFaces, nEdges);
  A.setFromTriplets(triplets.begin(), triplets.end());
}

// Loop rules for interior even edges
void BaseLoop::AssembleEdgeEvenBoundary(int edge, int vertInEdge, TripletInserter out) const {
  int row = GetEdgeEdgeIndex(edge, vertInEdge);
  int vert = _mesh->GetEdgeVerts(edge)[vertInEdge];
  int rSign = (GetVertVertIndex(vert) < GetEdgeVertIndex(edge)) ? 1 : -1;

  int nEdge = _mesh->GetVertEdges(vert).front();
  int vertInNedge = _mesh->GetVertIndexInEdge(nEdge, vert);
  int nSign = _mesh->GetVertSignInEdge(nEdge, vertInNedge);

  int pEdge = _mesh->GetVertEdges(vert).back();
  int vertInPedge = _mesh->GetVertIndexInEdge(pEdge, vert);
  int pSign = _mesh->GetVertSignInEdge(pEdge, vertInPedge);

  assert(edge == nEdge || edge == pEdge);

  if (edge == nEdge) {
    if (_isFixBnd) {
      // Fig1 even boundary rules in S.I. of [Chen et al. 2023]
      *out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.5 : 0.5);
    } else {
      // Fig2 even boundary rules in S.I. of [de Goes et al. 2016]
      *out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.375 : 0.375);
      *out++ = TripletX(row, pEdge, (pSign == rSign) ? 0.125 : -0.125);
    }

  } else {
    if (_isFixBnd) {
      // Fig1 even boundary rules in S.I. of [Chen et al. 2023]
      *out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.5 : 0.5);
    } else {
      // Fig2 even boundary rules in S.I. of [de Goes et al. 2016]
      *out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.375 : 0.375);
      *out++ = TripletX(row, nEdge, (nSign == rSign) ? 0.125 : -0.125);
    }
  }
}

// Insert Triplet from edge-edge case for one-form subdivision
void BaseLoop::InsertEdgeEdgeValue(int row, int col, int vert, int rSign, Scalar val, TripletInserter out) const {
  // Handy function that sets the sign of val for an edge col incident to vert.
  int vertInCol = _mesh->GetVertIndexInEdge(col, vert);
  int sign = _mesh->GetVertSignInEdge(col, vertInCol);
  *out++ = TripletX(row, col, (sign == rSign) ? -val : val);
}

// Insert Triplet from face-edge case for one-form subdivision
void BaseLoop::InsertEdgeFaceValue(int row, int face, int vert, int rSign, Scalar val, TripletInserter out) const {
  // Handy function that sets the sign of val for an edge col incident to face.
  int vertInFace = _mesh->GetVertIndexInFace(face, vert);
  int colInFace = (vertInFace + 1) % 3;
  int col = _mesh->GetFaceEdges(face)[colInFace];
  int sign = _mesh->GetEdgeSignInFace(face, colInFace);
  *out++ = TripletX(row, col, (sign == rSign) ? val : -val);
}

// Loop rules for boundary even edges (its endpoints are all on the boundary)
void BaseLoop::AssembleEdgeEvenPartialBoundary(int edge, int vertInEdge, TripletInserter out) const {
  int row = GetEdgeEdgeIndex(edge, vertInEdge);
  int vert = _mesh->GetEdgeVerts(edge)[vertInEdge];
  int rSign = (GetVertVertIndex(vert) < GetEdgeVertIndex(edge)) ? 1 : -1;

  const std::vector<int>& vEdges = _mesh->GetVertEdges(vert);
  const int edgeCount = vEdges.size();

  const std::vector<int>& vFaces = _mesh->GetVertFaces(vert);
  const int faceCount = vFaces.size();

  int edgeInVert = _mesh->GetEdgeIndexInVert(vert, edge);
  assert(edgeInVert != edgeCount - 1);
  assert(edgeInVert != 0);

  std::vector<std::pair<int, Scalar>> eValues;
  std::vector<std::pair<int, Scalar>> fValues;

  assert(faceCount > 1);

  if (faceCount == 2) {
    if (_isFixBnd) {
      // Fig1 even boundary adjacent rules top-left in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[0], 0.53125 / 3));
      eValues.push_back(std::make_pair(vEdges[1], 0.8125 / 3));
      eValues.push_back(std::make_pair(vEdges[2], 0.53125 / 3));
    } else {
      // Fig2 even boundary adjacent rules top-left in S.I. of [de Goes et al. 2016]
      eValues.push_back(std::make_pair(vEdges[0], 0.15625 / 3));
      eValues.push_back(std::make_pair(vEdges[1], 0.8125 / 3));
      eValues.push_back(std::make_pair(vEdges[2], 0.15625 / 3));
    }

    fValues.push_back(std::make_pair(vFaces[0], 0.15625 / 3));
    fValues.push_back(std::make_pair(vFaces[1], -0.15625 / 3));
  } else if (faceCount == 3 && edgeInVert == 1) {
    if (_isFixBnd) {
      // Fig1 even boundary adjacent rules top-mid in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[0], 0.53125 / 3));
      eValues.push_back(std::make_pair(vEdges[1], 0.3125));
      eValues.push_back(std::make_pair(vEdges[2], 0.09375));
      eValues.push_back(std::make_pair(vEdges[3], 0.125 / 3));
    } else {
      // Fig2 even boundary adjacent rules top-mid in S.I. of [de Goes et al. 2016]
      eValues.push_back(std::make_pair(vEdges[0], 0.15625 / 3));
      eValues.push_back(std::make_pair(vEdges[1], 0.3125));
      eValues.push_back(std::make_pair(vEdges[2], 0.09375));
      eValues.push_back(std::make_pair(vEdges[3], -0.25 / 3));
    }

    fValues.push_back(std::make_pair(vFaces[0], 0.15625 / 3));
    fValues.push_back(std::make_pair(vFaces[1], -0.03125 / 3));
    fValues.push_back(std::make_pair(vFaces[2], -0.125 / 3));
  } else if (faceCount == 3 && edgeInVert == 2) {
    if (_isFixBnd) {
      // Symmetric case of Fig1 even boundary adjacent rules top-mid in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[3], 0.53125 / 3));
      eValues.push_back(std::make_pair(vEdges[2], 0.3125));
      eValues.push_back(std::make_pair(vEdges[1], 0.09375));
      eValues.push_back(std::make_pair(vEdges[0], 0.125 / 3));
    } else {
      // Symmetric case of Fig2 even boundary adjacent rules top-mid in S.I. of [de Goes et al. 2016]
      eValues.push_back(std::make_pair(vEdges[3], 0.15625 / 3));
      eValues.push_back(std::make_pair(vEdges[2], 0.3125));
      eValues.push_back(std::make_pair(vEdges[1], 0.09375));
      eValues.push_back(std::make_pair(vEdges[0], -0.25 / 3));
    }

    fValues.push_back(std::make_pair(vFaces[2], -0.15625 / 3));
    fValues.push_back(std::make_pair(vFaces[1], 0.03125 / 3));
    fValues.push_back(std::make_pair(vFaces[0], 0.125 / 3));
  } else if (faceCount == 4 && edgeInVert == 2) {
    if (_isFixBnd) {
      // Fig1 even boundary adjacent rules bot-left in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[0], 0.03125));
      eValues.push_back(std::make_pair(vEdges[1], 0.125));
      eValues.push_back(std::make_pair(vEdges[2], 0.3125));
      eValues.push_back(std::make_pair(vEdges[3], 0.125));
      eValues.push_back(std::make_pair(vEdges[4], 0.03125));
    } else {
      // Fig2 even boundary adjacent rules bot-left in S.I. of [de Goes et al. 2016]
      eValues.push_back(std::make_pair(vEdges[0], -0.09375));
      eValues.push_back(std::make_pair(vEdges[1], 0.125));
      eValues.push_back(std::make_pair(vEdges[2], 0.3125));
      eValues.push_back(std::make_pair(vEdges[3], 0.125));
      eValues.push_back(std::make_pair(vEdges[4], -0.09375));
    }

    fValues.push_back(std::make_pair(vFaces[0], 0.03125));
    fValues.push_back(std::make_pair(vFaces[1], 0.03125));
    fValues.push_back(std::make_pair(vFaces[2], -0.03125));
    fValues.push_back(std::make_pair(vFaces[3], -0.03125));
  } else if (edgeInVert == 1) {
    if (_isFixBnd) {
      // Fig1 even boundary adjacent rules top-right in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[0], 0.53125 / 3));
      eValues.push_back(std::make_pair(vEdges[1], 0.3125));
      eValues.push_back(std::make_pair(vEdges[2], 0.09375));
      eValues.push_back(std::make_pair(vEdges[3], 0.125 / 3));
    } else {
      // Fig2 even boundary adjacent rules top-right in S.I. of [de Goes et al. 2016]
      eValues.push_back(std::make_pair(vEdges[0], 0.15625 / 3));
      eValues.push_back(std::make_pair(vEdges[1], 0.3125));
      eValues.push_back(std::make_pair(vEdges[2], 0.09375));
      eValues.push_back(std::make_pair(vEdges[3], 0.125 / 3));
      eValues.push_back(std::make_pair(vEdges.back(), -0.125));
    }

    fValues.push_back(std::make_pair(vFaces[0], 0.15625 / 3));
    fValues.push_back(std::make_pair(vFaces[1], -0.03125 / 3));
    fValues.push_back(std::make_pair(vFaces[2], -0.125 / 3));
  } else if (edgeInVert == edgeCount - 2) {
    if (_isFixBnd) {
      // Symmetric case of Fig1 even boundary adjacent rules top-right in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[edgeCount - 1], 0.53125 / 3));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 2], 0.3125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 3], 0.09375));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 4], 0.125 / 3));
    } else {
      // Symmetric case of Fig2 even boundary adjacent rules top-right in S.I. of [de Goes et al. 2016]
      eValues.push_back(std::make_pair(vEdges[edgeCount - 1], 0.15625 / 3));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 2], 0.3125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 3], 0.09375));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 4], 0.125 / 3));
      eValues.push_back(std::make_pair(vEdges.front(), -0.125));
    }

    fValues.push_back(std::make_pair(vFaces[faceCount - 1], -0.15625 / 3));
    fValues.push_back(std::make_pair(vFaces[faceCount - 2], 0.03125 / 3));
    fValues.push_back(std::make_pair(vFaces[faceCount - 3], 0.125 / 3));
  } else if (edgeInVert == 2) {
    if (_isFixBnd) {
      // Fig1 even boundary adjacent rules bot-mid in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[0], 0.03125));
      eValues.push_back(std::make_pair(vEdges[1], 0.125));
      eValues.push_back(std::make_pair(vEdges[2], 0.3125));
      eValues.push_back(std::make_pair(vEdges[3], 0.125));
      eValues.push_back(std::make_pair(vEdges[4], 0.03125));
    } else {
      // Fig2 even boundary adjacent rules bot-mid in S.I. of [de Goes et al. 2016]
      eValues.push_back(std::make_pair(vEdges[0], -0.09375));
      eValues.push_back(std::make_pair(vEdges[1], 0.125));
      eValues.push_back(std::make_pair(vEdges[2], 0.3125));
      eValues.push_back(std::make_pair(vEdges[3], 0.125));
      eValues.push_back(std::make_pair(vEdges[4], 0.03125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 1], -0.125));
    }


    fValues.push_back(std::make_pair(vFaces[0], 0.03125));
    fValues.push_back(std::make_pair(vFaces[1], 0.03125));
    fValues.push_back(std::make_pair(vFaces[2], -0.03125));
    fValues.push_back(std::make_pair(vFaces[3], -0.03125));
  } else if (edgeInVert == edgeCount - 3) {
    if (_isFixBnd) {
      // Symmetric case of Fig1 even boundary adjacent rules bot-mid in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[edgeCount - 1], 0.03125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 2], 0.125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 3], 0.3125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 4], 0.125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 5], 0.03125));
    } else {
      // Symmetric case of Fig2 even boundary adjacent rules bot-mid in S.I. of [de Goes et al. 2016]
      eValues.push_back(std::make_pair(vEdges[edgeCount - 1], -0.09375));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 2], 0.125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 3], 0.3125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 4], 0.125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 5], 0.03125));
      eValues.push_back(std::make_pair(vEdges[0], -0.125));
    }


    fValues.push_back(std::make_pair(vFaces[faceCount - 1], -0.03125));
    fValues.push_back(std::make_pair(vFaces[faceCount - 2], -0.03125));
    fValues.push_back(std::make_pair(vFaces[faceCount - 3], 0.03125));
    fValues.push_back(std::make_pair(vFaces[faceCount - 4], 0.03125));
  } else {
    if (_isFixBnd) {
      // Fig1 even boundary adjacent rules bot-right in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[edgeInVert - 2], 0.03125));
      eValues.push_back(std::make_pair(vEdges[edgeInVert - 1], 0.125));
      eValues.push_back(std::make_pair(vEdges[edgeInVert], 0.3125));
      eValues.push_back(std::make_pair(vEdges[edgeInVert + 1], 0.125));
      eValues.push_back(std::make_pair(vEdges[edgeInVert + 2], 0.03125));
    } else {
      // Fig2 even boundary adjacent rules bot-right in S.I. of [de Goes et al. 2016]
      eValues.push_back(std::make_pair(vEdges[edgeInVert - 2], 0.03125));
      eValues.push_back(std::make_pair(vEdges[edgeInVert - 1], 0.125));
      eValues.push_back(std::make_pair(vEdges[edgeInVert], 0.3125));
      eValues.push_back(std::make_pair(vEdges[edgeInVert + 1], 0.125));
      eValues.push_back(std::make_pair(vEdges[edgeInVert + 2], 0.03125));
      eValues.push_back(std::make_pair(vEdges.front(), -0.125));
      eValues.push_back(std::make_pair(vEdges.back(), -0.125));
    }


    fValues.push_back(std::make_pair(vFaces[edgeInVert - 2], 0.03125));
    fValues.push_back(std::make_pair(vFaces[edgeInVert - 1], 0.03125));
    fValues.push_back(std::make_pair(vFaces[edgeInVert], -0.03125));
    fValues.push_back(std::make_pair(vFaces[edgeInVert + 1], -0.03125));
  }

  for (size_t i = 0; i < eValues.size(); ++i) {
    InsertEdgeEdgeValue(row, eValues[i].first, vert, rSign, eValues[i].second, out);
  }

  for (size_t i = 0; i < fValues.size(); ++i) {
    InsertEdgeFaceValue(row, fValues[i].first, vert, rSign, fValues[i].second, out);
  }
}

// Loop rules for interior even edges
void BaseLoop::AssembleEdgeEvenInterior(int edge, int vertInEdge, TripletInserter out) const {
  int vert = _mesh->GetEdgeVerts(edge)[vertInEdge];
  int edgeInVert = _mesh->GetEdgeIndexInVert(vert, edge);

  int row = GetEdgeEdgeIndex(edge, vertInEdge);
  int rSign = (GetVertVertIndex(vert) < GetEdgeVertIndex(edge)) ? 1 : -1;

  const std::vector<int>& vEdges = _mesh->GetVertEdges(vert);
  const std::vector<int>& vFaces = _mesh->GetVertFaces(vert);

  std::vector<std::pair<int, Scalar>> eValues;
  std::vector<std::pair<int, Scalar>> fValues;

  const int count = vEdges.size();
  assert(count == vFaces.size());

  Scalar alpha = GetAlpha(vert);
  Scalar beta = GetBeta(vert);

  if (count == 3) {
    // Fig2 even interior rules bot in S.I. of [de Goes et al. 2016]
    eValues.push_back(std::make_pair(vEdges[(edgeInVert + 0) % count], 0.375 - alpha - 0.25 * beta));
    eValues.push_back(std::make_pair(vEdges[(edgeInVert + 1) % count], 0.125 * (1. + beta) - alpha));
    eValues.push_back(std::make_pair(vEdges[(edgeInVert + 2) % count], 0.125 * (1. + beta) - alpha));
  } else if (count == 4) {
    // Fig2 even interior rules mid in S.I. of [de Goes et al. 2016]
    eValues.push_back(std::make_pair(vEdges[(edgeInVert + 0) % count], 0.375 - alpha - 0.25 * beta));
    eValues.push_back(std::make_pair(vEdges[(edgeInVert + 1) % count], 0.125 - alpha));
    eValues.push_back(std::make_pair(vEdges[(edgeInVert + 2) % count], 0.25 * beta - alpha));
    eValues.push_back(std::make_pair(vEdges[(edgeInVert + 3) % count], 0.125 - alpha));
  } else {
    // Fig2 even interior rules top in S.I. of [de Goes et al. 2016]
    for (int i = 0; i < count; ++i) {
      Scalar val = 0.;
      if (i == 0)
        val = 0.375 - alpha - 0.25 * beta;
      else if (i == 1 || i == count - 1)
        val = 0.125 - alpha;
      else if (i == 2 || i == count - 2)
        val = 0.125 * beta - alpha;
      else
        val = -alpha;
      eValues.push_back(std::make_pair(vEdges[(edgeInVert + i) % count], val));
    }
  }

  if (count == 3) {
    // Fig2 even interior rules bot in S.I. of [de Goes et al. 2016]
    fValues.push_back(std::make_pair(vFaces[(edgeInVert + 0) % count], -0.125 * beta));
    fValues.push_back(std::make_pair(vFaces[(edgeInVert + 2) % count], 0.125 * beta));
  } else {
    // Fig2 even interior rules top/mid in S.I. of [de Goes et al. 2016]
    fValues.push_back(std::make_pair(vFaces[(edgeInVert + 0) % count], -0.125 * beta));
    fValues.push_back(std::make_pair(vFaces[(edgeInVert + 1) % count], -0.125 * beta));
    fValues.push_back(std::make_pair(vFaces[(edgeInVert + count - 1) % count], 0.125 * beta));
    fValues.push_back(std::make_pair(vFaces[(edgeInVert + count - 2) % count], 0.125 * beta));
  }

  for (size_t i = 0; i < eValues.size(); ++i) {
    InsertEdgeEdgeValue(row, eValues[i].first, vert, rSign, eValues[i].second, out);
  }

  for (size_t i = 0; i < fValues.size(); ++i) {
    InsertEdgeFaceValue(row, fValues[i].first, vert, rSign, fValues[i].second, out);
  }
}

// Loop rules for odd edges
void BaseLoop::AssembleEdgeOdd(int face, int edgeInFace, TripletInserter out) const {
  int row = GetFaceEdgeIndex(face, edgeInFace);

  int vertInFace = (edgeInFace + 1) % 3;
  int vert = _mesh->GetFaceVerts(face)[vertInFace];

  int nEdge = _mesh->GetFaceEdges(face)[vertInFace];
  int nSign = _mesh->GetEdgeSignInFace(face, vertInFace);

  int oEdge = _mesh->GetFaceEdges(face)[(vertInFace + 1) % 3];
  int oSign = _mesh->GetEdgeSignInFace(face, (vertInFace + 1) % 3);

  int pEdge = _mesh->GetFaceEdges(face)[(vertInFace + 2) % 3];
  int pSign = _mesh->GetEdgeSignInFace(face, (vertInFace + 2) % 3);

  int rSign = (GetEdgeVertIndex(nEdge) < GetEdgeVertIndex(pEdge)) ? 1 : -1;

  bool nBdry = _mesh->IsEdgeBoundary(nEdge);
  bool pBdry = _mesh->IsEdgeBoundary(pEdge);

  if (nBdry && pBdry) {
    // Fig1 top middle in S.I. of [Chen et al. 2023]
    *out++ = TripletX(row, oEdge, (oSign == rSign) ? 0.25 : -0.25);
    *out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.25 : 0.25);
    *out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.25 : 0.25);
  } else if (nBdry) {
    // Symmetric case of Fig2 odd boundary adjacent rules top-left in S.I. of [de Goes et al. 2016]
    *out++ = TripletX(row, oEdge, (oSign == rSign) ? 0.21875 : -0.21875);
    *out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.1875 : 0.1875);
    *out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.15625 : 0.15625);
  } else if (pBdry) {
    // Fig2 odd boundary adjacent rules top-left in S.I. of [de Goes et al. 2016]
    *out++ = TripletX(row, oEdge, (oSign == rSign) ? 0.21875 : -0.21875);
    *out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.15625 : 0.15625);
    *out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.1875 : 0.1875);
  } else {
    // Fig2 odd interior rules top-left in S.I. of [de Goes et al. 2016]
    *out++ = TripletX(row, oEdge, (oSign == rSign) ? 0.1875 : -0.1875);
    *out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.09375 : 0.09375);
    *out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.09375 : 0.09375);
  }

  std::vector<int> flaps;
  flaps.push_back(pEdge);
  flaps.push_back(nEdge);
  for (int i = 0; i < flaps.size(); ++i) {
    int edge = flaps[i];
    if (_mesh->IsEdgeBoundary(edge)) continue;

    int faceInEdge = _mesh->GetFaceIndexInEdge(edge, face);
    int flap = _mesh->GetEdgeFaces(edge)[(faceInEdge + 1) % 2];
    int edgeInFlap = _mesh->GetEdgeIndexInFace(flap, edge);

    for (int j = 1; j <= 2; ++j) {
      // Fig2 odd interior rules top-left in S.I. of [de Goes et al. 2016]
      Scalar val = 0.09375;
      int cEdge = _mesh->GetFaceEdges(flap)[(edgeInFlap + j) % 3];
      int cSign = _mesh->GetEdgeSignInFace(flap, (edgeInFlap + j) % 3);
      if (_mesh->GetVertIndexInEdge(cEdge, vert) == -1) val = -0.03125;
      *out++ = TripletX(row, cEdge, (cSign == rSign) ? -val : val);
    }
  }
}
} // namespace ComplexWrinkleField
