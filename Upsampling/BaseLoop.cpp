#include "BaseLoop.h"
#include "../CommonTools.h"
#include <cassert>
#include <iostream>
#include <memory>

namespace ComplexWrinkleField {
bool BaseLoop::IsVertRegular(int vert) const {
  if (_mesh->IsVertBoundary(vert)) return true;
  return (_mesh->GetVertEdges(vert).size() == 6);
}

bool BaseLoop::AreIrregularVertsIsolated() const {
  for (int edge = 0; edge < _mesh->GetEdgeCount(); ++edge) {
    const std::vector<int>& eVerts = _mesh->GetEdgeVerts(edge);
    if (IsVertRegular(eVerts[0])) continue;
    if (IsVertRegular(eVerts[1])) continue;
    return false;
  }
  return true;
}

int BaseLoop::_GetVertVertIndex(int vert) const { return vert; }

int BaseLoop::_GetEdgeVertIndex(int edge) const { return _mesh->GetVertCount() + edge; }

int BaseLoop::_GetEdgeEdgeIndex(int edge, int vertInEdge) const { return 2 * edge + vertInEdge; }

int BaseLoop::_GetFaceEdgeIndex(int face, int edgeInFace) const {

  assert(_mesh->IsTriangulated());
  return 2 * _mesh->GetEdgeCount() + 3 * face + edgeInFace;
}

int BaseLoop::_GetCentralFaceIndex(int face) const {

  assert(_mesh->IsTriangulated());
  return 4 * face + 3;
}

int BaseLoop::_GetCornerFaceIndex(int face, int vertInFace) const {

  assert(_mesh->IsTriangulated());
  return 4 * face + vertInFace;
}

void BaseLoop::GetSubdividedEdges(std::vector<std::vector<int>>& edgeToVert) const {

  assert(_mesh->IsTriangulated());

  int E = _mesh->GetEdgeCount();
  int F = _mesh->GetFaceCount();
  edgeToVert.resize(2 * E + 3 * F);

  for (int edge = 0; edge < E; ++edge) {
    const std::vector<int>& eVerts = _mesh->GetEdgeVerts(edge);
    for (int i = 0; i < 2; ++i) {
      int v0 = _GetVertVertIndex(eVerts[i]);
      int v1 = _GetEdgeVertIndex(edge);
      if (v0 > v1) std::swap(v0, v1);

      int index = _GetEdgeEdgeIndex(edge, i);
      edgeToVert[index].push_back(v0);
      edgeToVert[index].push_back(v1);
    }
  }

  for (int face = 0; face < F; ++face) {
    const std::vector<int>& fEdges = _mesh->GetFaceEdges(face);
    for (int i = 0; i < 3; ++i) {
      int v0 = _GetEdgeVertIndex(fEdges[i]);
      int v1 = _GetEdgeVertIndex(fEdges[(i + 1) % 3]);
      if (v0 > v1) std::swap(v0, v1);

      int index = _GetFaceEdgeIndex(face, i);
      edgeToVert[index].push_back(v0);
      edgeToVert[index].push_back(v1);
    }
  }
}

void BaseLoop::GetSubdividedFaces(std::vector<std::vector<int>>& faceToVert) const {

  assert(_mesh->IsTriangulated());

  int V = _mesh->GetVertCount();
  int F = _mesh->GetFaceCount();
  faceToVert.resize(4 * F);

  std::vector<int> faceFlagsNew;
  faceFlagsNew.resize(4 * F, 0);

  for (int face = 0; face < _mesh->GetFaceCount(); ++face) {
    int central = _GetCentralFaceIndex(face);
    const std::vector<int>& fVerts = _mesh->GetFaceVerts(face);
    const std::vector<int>& fEdges = _mesh->GetFaceEdges(face);
    for (int j = 0; j < 3; ++j) {
      // Corner face
      int index = _GetCornerFaceIndex(face, j);
      faceToVert[index].push_back(_GetVertVertIndex(fVerts[j]));
      faceToVert[index].push_back(_GetEdgeVertIndex(fEdges[j]));
      faceToVert[index].push_back(_GetEdgeVertIndex(fEdges[(j + 2) % 3]));
      // Central face
      faceToVert[central].push_back(_GetEdgeVertIndex(fEdges[j]));
    }
  }
}

Scalar BaseLoop::_GetAlpha(int vert) const {

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

Scalar BaseLoop::_GetBeta(int vert) const {

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

void BaseLoop::BuildS0(SparseMatrixX& A) const {
  assert(_mesh && _mesh->IsTriangulated());

  int nrows, ncols;
  GetS0Size(nrows, ncols);

  std::vector<TripletX> triplets;
  const int nverts = _mesh->GetVertCount();
  const int nedges = _mesh->GetEdgeCount();

  // Even (old) verts
  for (int vi = 0; vi < nverts; ++vi) {
    if (_mesh->IsVertBoundary(vi))
      _AssembleVertEvenBoundary(vi, std::back_inserter(triplets));
    else
      _AssembleVertEvenInterior(vi, std::back_inserter(triplets));
  }

  // Odd (new) verts
  for (int edge = 0; edge < nedges; ++edge) {
    if (_mesh->IsEdgeBoundary(edge))
      _AssembleVertOddBoundary(edge, std::back_inserter(triplets));
    else
      _AssembleVertOddInterior(edge, std::back_inserter(triplets));
  }

  A.resize(nrows, ncols);
  A.setFromTriplets(triplets.begin(), triplets.end());
}

void BaseLoop::BuildS1(SparseMatrixX& A) const {

  assert(_mesh->IsTriangulated());

  std::vector<TripletX> triplets;
  const int nedges = _mesh->GetEdgeCount();
  const int nfaces = _mesh->GetFaceCount();

  for (int edge = 0; edge < nedges; ++edge) {
    if (_mesh->IsEdgeBoundary(edge)) {
      _AssembleEdgeEvenBoundary(edge, 0, std::back_inserter(triplets));
      _AssembleEdgeEvenBoundary(edge, 1, std::back_inserter(triplets));
    } else {
      const std::vector<int>& eVerts = _mesh->GetEdgeVerts(edge);
      for (int i = 0; i < eVerts.size(); ++i) {
        if (_mesh->IsVertBoundary(eVerts[i]))
          _AssembleEdgeEvenPartialBoundary(edge, i, std::back_inserter(triplets));
        else
          _AssembleEdgeEvenInterior(edge, i, std::back_inserter(triplets));
      }
    }
  }

  for (int face = 0; face < nfaces; ++face) {
    _AssembleEdgeOdd(face, 0, std::back_inserter(triplets));
    _AssembleEdgeOdd(face, 1, std::back_inserter(triplets));
    _AssembleEdgeOdd(face, 2, std::back_inserter(triplets));
  }

  A.resize(2 * nedges + 3 * nfaces, nedges);
  A.setFromTriplets(triplets.begin(), triplets.end());
}

void BaseLoop::_AssembleEdgeEvenBoundary(int edge, int vertInEdge, TripletInserter out) const {
  int row = _GetEdgeEdgeIndex(edge, vertInEdge);
  int vert = _mesh->GetEdgeVerts(edge)[vertInEdge];
  int rSign = (_GetVertVertIndex(vert) < _GetEdgeVertIndex(edge)) ? 1 : -1;

  int nEdge = _mesh->GetVertEdges(vert).front();
  int vertInNedge = _mesh->GetVertIndexInEdge(nEdge, vert);
  int nSign = _mesh->GetVertSignInEdge(nEdge, vertInNedge);

  int pEdge = _mesh->GetVertEdges(vert).back();
  int vertInPedge = _mesh->GetVertIndexInEdge(pEdge, vert);
  int pSign = _mesh->GetVertSignInEdge(pEdge, vertInPedge);

  assert(edge == nEdge || edge == pEdge);

  if (edge == nEdge) {
    if (_isFixBnd) {
      // Fig1 even boudary rules in S.I. of [Chen et al. 2023]
      *out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.5 : 0.5);
    } else {
      // Fig2 even boudary rules in S.I. of [de Goes et al. 2016]
      *out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.375 : 0.375);
      *out++ = TripletX(row, pEdge, (pSign == rSign) ? 0.125 : -0.125);
    }

  } else {
    if (_isFixBnd) {
      // Fig1 even boudary rules in S.I. of [Chen et al. 2023]
      *out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.5 : 0.5);
    } else {
      // Fig2 even boudary rules in S.I. of [de Goes et al. 2016]
      *out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.375 : 0.375);
      *out++ = TripletX(row, nEdge, (nSign == rSign) ? 0.125 : -0.125);
    }
  }
}

void BaseLoop::_InsertEdgeEdgeValue(int row, int col, int vert, int rSign, Scalar val, TripletInserter out) const {
  // Handy function that sets the sign of val for an edge col incident to vert.
  int vertInCol = _mesh->GetVertIndexInEdge(col, vert);
  int sign = _mesh->GetVertSignInEdge(col, vertInCol);
  *out++ = TripletX(row, col, (sign == rSign) ? -val : val);
}

void BaseLoop::_InsertEdgeFaceValue(int row, int face, int vert, int rSign, Scalar val, TripletInserter out) const {
  // Handy function that sets the sign of val for an edge col incident to face.
  int vertInFace = _mesh->GetVertIndexInFace(face, vert);
  int colInFace = (vertInFace + 1) % 3;
  int col = _mesh->GetFaceEdges(face)[colInFace];
  int sign = _mesh->GetEdgeSignInFace(face, colInFace);
  *out++ = TripletX(row, col, (sign == rSign) ? val : -val);
}

void BaseLoop::_AssembleEdgeEvenPartialBoundary(int edge, int vertInEdge, TripletInserter out) const {
  int row = _GetEdgeEdgeIndex(edge, vertInEdge);
  int vert = _mesh->GetEdgeVerts(edge)[vertInEdge];
  int rSign = (_GetVertVertIndex(vert) < _GetEdgeVertIndex(edge)) ? 1 : -1;

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
      // Fig1 even boudary adjacent rules top-left in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[0], 0.53125 / 3));
      eValues.push_back(std::make_pair(vEdges[1], 0.8125 / 3));
      eValues.push_back(std::make_pair(vEdges[2], 0.53125 / 3));
    } else {
      // Fig2 even boudary adjacent rules top-left in S.I. of [de Goes et al. 2016]
      eValues.push_back(std::make_pair(vEdges[0], 0.15625 / 3));
      eValues.push_back(std::make_pair(vEdges[1], 0.8125 / 3));
      eValues.push_back(std::make_pair(vEdges[2], 0.15625 / 3));
    }


    fValues.push_back(std::make_pair(vFaces[0], 0.15625 / 3));
    fValues.push_back(std::make_pair(vFaces[1], -0.15625 / 3));
  } else if (faceCount == 3 && edgeInVert == 1) {
    if (_isFixBnd) {
      // Fig1 even boudary adjacent rules top-mid in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[0], 0.53125 / 3));
      eValues.push_back(std::make_pair(vEdges[1], 0.3125));
      eValues.push_back(std::make_pair(vEdges[2], 0.09375));
      eValues.push_back(std::make_pair(vEdges[3], 0.125 / 3));
    } else {
      // Fig2 even boudary adjacent rules top-mid in S.I. of [de Goes et al. 2016]
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
      // Symmetric case of Fig1 even boudary adjacent rules top-mid in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[3], 0.53125 / 3));
      eValues.push_back(std::make_pair(vEdges[2], 0.3125));
      eValues.push_back(std::make_pair(vEdges[1], 0.09375));
      eValues.push_back(std::make_pair(vEdges[0], 0.125 / 3));
    } else {
      // Symmetric case of Fig2 even boudary adjacent rules top-mid in S.I. of [de Goes et al. 2016]
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
      // Fig1 even boudary adjacent rules bot-left in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[0], 0.03125));
      eValues.push_back(std::make_pair(vEdges[1], 0.125));
      eValues.push_back(std::make_pair(vEdges[2], 0.3125));
      eValues.push_back(std::make_pair(vEdges[3], 0.125));
      eValues.push_back(std::make_pair(vEdges[4], 0.03125));
    } else {
      // Fig2 even boudary adjacent rules bot-left in S.I. of [de Goes et al. 2016]
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
      // Fig1 even boudary adjacent rules top-right in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[0], 0.53125 / 3));
      eValues.push_back(std::make_pair(vEdges[1], 0.3125));
      eValues.push_back(std::make_pair(vEdges[2], 0.09375));
      eValues.push_back(std::make_pair(vEdges[3], 0.125 / 3));

      fValues.push_back(std::make_pair(vFaces[0], 0.15625 / 3));
      fValues.push_back(std::make_pair(vFaces[1], -0.03125 / 3));
      fValues.push_back(std::make_pair(vFaces[2], -0.125 / 3));
    } else {
      // Fig2 even boudary adjacent rules top-right in S.I. of [de Goes et al. 2016]
      eValues.push_back(std::make_pair(vEdges[0], 0.15625 / 3));
      eValues.push_back(std::make_pair(vEdges[1], 0.3125)); 
      eValues.push_back(std::make_pair(vEdges[2], 0.09375));
      eValues.push_back(std::make_pair(vEdges[3], 0.125 / 3));
      eValues.push_back(std::make_pair(vEdges.back(), -0.125));
    }
  } else if (edgeInVert == edgeCount - 2) {
    if (_isFixBnd) {
      // Symmetric case of Fig1 even boudary adjacent rules top-right in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[edgeCount - 1], 0.53125 / 3));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 2], 0.3125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 3], 0.09375));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 4], 0.125 / 3));
    } else {
      // Symmetric case of Fig2 even boudary adjacent rules top-right in S.I. of [de Goes et al. 2016]
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
      // Fig1 even boudary adjacent rules bot-mid in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[0], 0.03125));
      eValues.push_back(std::make_pair(vEdges[1], 0.125));
      eValues.push_back(std::make_pair(vEdges[2], 0.3125));
      eValues.push_back(std::make_pair(vEdges[3], 0.125));
      eValues.push_back(std::make_pair(vEdges[4], 0.03125));
    } else {
      // Fig2 even boudary adjacent rules bot-mid in S.I. of [de Goes et al. 2016]
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
      // Symmetric case of Fig1 even boudary adjacent rules bot-mid in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[edgeCount - 1], 0.03125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 2], 0.125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 3], 0.3125)); 
      eValues.push_back(std::make_pair(vEdges[edgeCount - 4], 0.125));
      eValues.push_back(std::make_pair(vEdges[edgeCount - 5], 0.03125));
    } else {
      // Symmetric case of Fig2 even boudary adjacent rules bot-mid in S.I. of [de Goes et al. 2016]
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
      // Fig1 even boudary adjacent rules bot-right in S.I. of [Chen et al. 2023]
      eValues.push_back(std::make_pair(vEdges[edgeInVert - 2], 0.03125));
      eValues.push_back(std::make_pair(vEdges[edgeInVert - 1], 0.125));
      eValues.push_back(std::make_pair(vEdges[edgeInVert], 0.3125)); 
      eValues.push_back(std::make_pair(vEdges[edgeInVert + 1], 0.125));
      eValues.push_back(std::make_pair(vEdges[edgeInVert + 2], 0.03125));
    } else {
      // Fig2 even boudary adjacent rules bot-right in S.I. of [de Goes et al. 2016]
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
    _InsertEdgeEdgeValue(row, eValues[i].first, vert, rSign, eValues[i].second, out);
  }

  for (size_t i = 0; i < fValues.size(); ++i) {
    _InsertEdgeFaceValue(row, fValues[i].first, vert, rSign, fValues[i].second, out);
  }
}

void BaseLoop::_AssembleEdgeEvenInterior(int edge, int vertInEdge, TripletInserter out) const {
  int vert = _mesh->GetEdgeVerts(edge)[vertInEdge];
  int edgeInVert = _mesh->GetEdgeIndexInVert(vert, edge);

  int row = _GetEdgeEdgeIndex(edge, vertInEdge);
  int rSign = (_GetVertVertIndex(vert) < _GetEdgeVertIndex(edge)) ? 1 : -1;

  const std::vector<int>& vEdges = _mesh->GetVertEdges(vert);
  const std::vector<int>& vFaces = _mesh->GetVertFaces(vert);

  std::vector<std::pair<int, Scalar>> eValues;
  std::vector<std::pair<int, Scalar>> fValues;

  const int count = vEdges.size();
  assert(count == vFaces.size());

  Scalar alpha = _GetAlpha(vert);
  Scalar beta = _GetBeta(vert);

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
    _InsertEdgeEdgeValue(row, eValues[i].first, vert, rSign, eValues[i].second, out);
  }

  for (size_t i = 0; i < fValues.size(); ++i) {
    _InsertEdgeFaceValue(row, fValues[i].first, vert, rSign, fValues[i].second, out);
  }
}

void BaseLoop::_AssembleEdgeOdd(int face, int edgeInFace, TripletInserter out) const {
  int row = _GetFaceEdgeIndex(face, edgeInFace);

  int vertInFace = (edgeInFace + 1) % 3;
  int vert = _mesh->GetFaceVerts(face)[vertInFace];

  int nEdge = _mesh->GetFaceEdges(face)[vertInFace];
  int nSign = _mesh->GetEdgeSignInFace(face, vertInFace);

  int oEdge = _mesh->GetFaceEdges(face)[(vertInFace + 1) % 3];
  int oSign = _mesh->GetEdgeSignInFace(face, (vertInFace + 1) % 3);

  int pEdge = _mesh->GetFaceEdges(face)[(vertInFace + 2) % 3];
  int pSign = _mesh->GetEdgeSignInFace(face, (vertInFace + 2) % 3);

  int rSign = (_GetEdgeVertIndex(nEdge) < _GetEdgeVertIndex(pEdge)) ? 1 : -1;

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
