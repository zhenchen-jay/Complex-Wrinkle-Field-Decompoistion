#include "ComplexLoop.h"

namespace ComplexWrinkleField {
/***** Loop subdivision for 0-form (per-vertex complex value, [Chen et al, 2023]) *****/
// Loop rules for interior even vertices
void ComplexLoop::AssembleVertEvenInterior(int vi, TripletInserter out) const {
  // Fig13(a) left in [Chen et al. 2023]
  const std::vector<int>& vFaces = _mesh->GetVertFaces(vi);
  int nNeiFaces = vFaces.size();

  Scalar alpha = GetAlpha(vi);
  int row = GetVertVertIndex(vi);

  Scalar beta = nNeiFaces / 2. * alpha;

  std::vector<std::complex<Scalar>> zp(nNeiFaces);
  std::vector<Vector3> gradthetap(nNeiFaces);
  std::vector<Scalar> coords;
  coords.resize(nNeiFaces, 1. / nNeiFaces);
  std::vector<Vector3> pList(nNeiFaces);
  std::vector<std::vector<std::complex<Scalar>>> innerWeights(nNeiFaces);
  std::vector<std::complex<Scalar>> outerWeights(nNeiFaces);
  std::vector<std::vector<int>> faceVertMap(nNeiFaces, {-1, -1, -1});

  for (int k = 0; k < nNeiFaces; ++k) {
    int face = vFaces[k];
    int viInface = _mesh->GetVertIndexInFace(face, vi);
    Vector3 bary;
    bary.setConstant(beta);
    bary(viInface) = 1 - 2 * beta;

    pList[k] = Vector3::Zero();
    for (int i = 0; i < 3; i++) {
      pList[k] += bary(i) * _mesh->GetVertPos(_mesh->GetFaceVerts(face)[i]);
    }
    faceVertMap[k] = _mesh->GetFaceVerts(face);

    innerWeights[k] = ComputeTriangleComplexWeight(bary, face);
    gradthetap[k] = ComputeBaryGradThetaFromOmegaPerface(face, bary);
  }
  outerWeights = ComputeComplexWeightFromGradTheta(pList, gradthetap, coords);

  for (int j = 0; j < nNeiFaces; j++) {
    for (int k = 0; k < 3; k++) {
      std::complex<Scalar> cjk = outerWeights[j] * innerWeights[j][k];
      FillTriplets(row, faceVertMap[j][k], _mesh->GetVertCount() + _mesh->GetEdgeCount(), _mesh->GetVertCount(), cjk,
                   out);
    }
  }
}

// Loop rules for boundary even vertices
void ComplexLoop::AssembleVertEvenBoundary(int vi, TripletInserter out) const {
  // Fig13(a) right in [Chen et al. 2023]
  int row = GetVertVertIndex(vi);
  if (_isFixBnd) {
    FillTriplets(row, vi, _mesh->GetVertCount() + _mesh->GetEdgeCount(), _mesh->GetVertCount(), 1.0, out);
  } else {
    std::vector<int> boundary(2);
    boundary[0] = _mesh->GetVertEdges(vi).front();
    boundary[1] = _mesh->GetVertEdges(vi).back();

    std::vector<Vector3> gradthetap(2);
    std::vector<Scalar> coords = {1. / 2, 1. / 2};
    std::vector<Vector3> pList(2);
    std::vector<std::vector<int>> edgeVertMap(2, {-1, -1});

    std::vector<std::vector<std::complex<Scalar>>> innerWeights(2);
    std::vector<std::complex<Scalar>> outerWeights(2);

    for (int j = 0; j < boundary.size(); ++j) {
      int edge = boundary[j];
      int face = _mesh->GetEdgeFaces(edge)[0];
      int viInface = _mesh->GetVertIndexInFace(face, vi);

      int viInEdge = _mesh->GetVertIndexInEdge(edge, vi);
      int vj = _mesh->GetEdgeVerts(edge)[(viInEdge + 1) % 2];

      int vjInface = _mesh->GetVertIndexInFace(face, vj);

      Vector3 bary = Vector3::Zero();
      bary(viInface) = 3. / 4;
      bary(vjInface) = 1. / 4;

      Vector2 edgeBary;
      edgeBary[viInEdge] = 3. / 4.;
      edgeBary[1 - viInEdge] = 1. / 4.;

      edgeVertMap[j][viInEdge] = vi;
      edgeVertMap[j][1 - viInEdge] = vj;

      pList[j] = 3. / 4 * _mesh->GetVertPos(vi) + 1. / 4 * _mesh->GetVertPos(vj);
      // grad from vi
      gradthetap[j] = ComputeBaryGradThetaFromOmegaPerface(face, bary);
      innerWeights[j] = ComputeEdgeComplexWeight(edgeBary, edge);
    }
    outerWeights = ComputeComplexWeightFromGradTheta(pList, gradthetap, coords);

    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        std::complex<Scalar> cjk = outerWeights[j] * innerWeights[j][k];
        FillTriplets(row, edgeVertMap[j][k], _mesh->GetVertCount() + _mesh->GetEdgeCount(), _mesh->GetVertCount(), cjk,
                     out);
      }
    }
  }
}

// Loop rules for interior odd vertices
void ComplexLoop::AssembleVertOddInterior(int edge, TripletInserter out) const {
  // Fig13(b) left in [Chen et al. 2023]
  std::vector<Vector3> gradthetap(2);
  std::vector<Scalar> coords = {1. / 2, 1. / 2};
  std::vector<Vector3> pList(2);
  std::vector<std::vector<std::complex<Scalar>>> innerWeights(2);
  std::vector<std::complex<Scalar>> outerWeights(2);
  std::vector<std::vector<int>> faceVertMap(2, {-1, -1, -1});

  int row = GetEdgeVertIndex(edge);

  for (int j = 0; j < 2; ++j) {
    int face = _mesh->GetEdgeFaces(edge)[j];
    int offset = _mesh->GetEdgeIndexInFace(face, edge);

    Vector3 bary;
    bary.setConstant(3. / 8.);
    bary((offset + 2) % 3) = 0.25;

    pList[j] = Vector3::Zero();
    for (int i = 0; i < 3; i++) {
      pList[j] += bary(i) * _mesh->GetVertPos(_mesh->GetFaceVerts(face)[i]);
    }
    faceVertMap[j] = _mesh->GetFaceVerts(face);
    innerWeights[j] = ComputeTriangleComplexWeight(bary, face);
    gradthetap[j] = ComputeBaryGradThetaFromOmegaPerface(face, bary);
  }
  outerWeights = ComputeComplexWeightFromGradTheta(pList, gradthetap, coords);

  for (int j = 0; j < 2; j++) {
    for (int k = 0; k < 3; k++) {
      std::complex<Scalar> cjk = outerWeights[j] * innerWeights[j][k];
      FillTriplets(row, faceVertMap[j][k], _mesh->GetVertCount() + _mesh->GetEdgeCount(), _mesh->GetVertCount(), cjk,
                   out);
    }
  }
}

// Loop rules for boundary odd vertices
void ComplexLoop::AssembleVertOddBoundary(int edge, TripletInserter out) const {
  // Fig13(b) right in [Chen et al. 2023]
  Vector2 bary;
  bary << 0.5, 0.5;
  int row = GetEdgeVertIndex(edge);
  std::vector<std::complex<Scalar>> complexWeight = ComputeEdgeComplexWeight(bary, edge);
  FillTriplets(row, _mesh->GetEdgeVerts(edge)[0], _mesh->GetVertCount() + _mesh->GetEdgeCount(), _mesh->GetVertCount(),
               complexWeight[0], out);
  FillTriplets(row, _mesh->GetEdgeVerts(edge)[1], _mesh->GetVertCount() + _mesh->GetEdgeCount(), _mesh->GetVertCount(),
               complexWeight[1], out);
}

// Barycentrically blend the z values from the corner of the polygon:
// the angle contribution from each point given by dthetaList[i]. Then z = \sum pWeight[i] * exp(dthetaList[i]) * z(p_i).
std::vector<std::complex<Scalar>> ComplexLoop::ComputeComplexWeight(const std::vector<Scalar>& dthetaList,
                                                                    const std::vector<Scalar>& coordList) const {
  // The generalization of Equation (19) in [Chen et al, 2023]
  int nPoints = dthetaList.size();
  std::vector<std::complex<Scalar>> complexWeights(nPoints, 0);

  for (int i = 0; i < nPoints; i++) {
    complexWeights[i] = coordList[i] * std::complex<Scalar>(std::cos(dthetaList[i]), std::sin(dthetaList[i]));
  }
  return complexWeights;
}

// Barycentrically blend the z values from the corner of the polygon:
// The new position is given as \sum pWeights[i] * pList[i], the angle contribution for the i-th point is:
// dtheta_i = gradThetaList[i].dot(p - pList[i]). Then z = \sum pWeight[i] * exp(dtheta_i) * z(p_i).
std::vector<std::complex<Scalar>> ComplexLoop::ComputeComplexWeightFromGradTheta(const std::vector<Vector3>& pList,
                                                                    const std::vector<Vector3>& gradThetaList,
                                                                    const std::vector<Scalar>& pWeights) const {
  // The generalization of Equation (19) in [Chen et al, 2023]
  int nPoints = pList.size();
  std::vector<Scalar> dthetaList(nPoints, 0);

  Vector3 p = Vector3::Zero();
  for (int i = 0; i < nPoints; i++) {
    p += pWeights[i] * pList[i];
  }

  for (int i = 0; i < nPoints; i++) {
    dthetaList[i] = gradThetaList[i].dot(p - pList[i]);
  }
  return ComputeComplexWeight(dthetaList, pWeights);
}

// Barycentrially blender the z value, where the new position point is on the edge
std::vector<std::complex<Scalar>> ComplexLoop::ComputeEdgeComplexWeight(const Vector2& bary, int eid) const {
  // Equation (18) in [Chen et al, 2023]
  Scalar edgeOmega = (*_omega)[eid]; // one form, refered as theta[e1] - theta[e0]
  Scalar dtheta0 = bary[1] * edgeOmega;
  Scalar dtheta1 = -bary[0] * edgeOmega;

  std::vector<Scalar> dthetaList = {dtheta0, dtheta1};
  std::vector<Scalar> coordList = {bary[0], bary[1]};

  return ComputeComplexWeight(dthetaList, coordList);
}

// Barycentrially blender the z value, where the new position point is on the face
std::vector<std::complex<Scalar>> ComplexLoop::ComputeTriangleComplexWeight(const Vector3& bary, int fid) const {
  // Equation (19) in [Chen et al, 2023]
  std::vector<Scalar> coordList = {bary[0], bary[1], bary[2]};
  std::vector<Scalar> dthetaList(3, 0);
  const std::vector<int>& vertList = _mesh->GetFaceVerts(fid);
  const std::vector<int>& edgeList = _mesh->GetFaceEdges(fid);

  for (int i = 0; i < 3; i++) {
    int vid = vertList[i];
    int eid0 = edgeList[i];
    int eid1 = edgeList[(i + 2) % 3];

    Scalar w0 = (*_omega)[eid0];
    Scalar w1 = (*_omega)[eid1];

    if (vid == _mesh->GetEdgeVerts(eid0)[1]) w0 *= -1;
    if (vid == _mesh->GetEdgeVerts(eid1)[1]) w1 *= -1;

    dthetaList[i] = w0 * bary((i + 1) % 3) + w1 * bary((i + 2) % 3);
  }
  return ComputeComplexWeight(dthetaList, coordList);
}

// For each vertex on the face, compute the approximate gradient of theta from edge 1-forms.
Vector3 ComplexLoop::ComputeGradThetaFromOmegaPerfaceCorner(int fid, int vInF) const {
  // Equation (52) in S.I.of [Chen et al, 2023]
  int eid0 = _mesh->GetFaceEdges(fid)[vInF];
  int eid1 = _mesh->GetFaceEdges(fid)[(vInF + 2) % 3];
  Vector3 r0 = _mesh->GetVertPos(_mesh->GetEdgeVerts(eid0)[1]) - _mesh->GetVertPos(_mesh->GetEdgeVerts(eid0)[0]);
  Vector3 r1 = _mesh->GetVertPos(_mesh->GetEdgeVerts(eid1)[1]) - _mesh->GetVertPos(_mesh->GetEdgeVerts(eid1)[0]);

  Eigen::Matrix2d Iinv, I;
  I << r0.dot(r0), r0.dot(r1), r1.dot(r0), r1.dot(r1);
  Iinv = I.inverse();

  Vector2 rhs;
  rhs << (*_omega)[eid0], (*_omega)[eid1];

  Vector2 u = Iinv * rhs;
  return u[0] * r0 + u[1] * r1;
}

// Barycentrically blend the gradient of theta from face corners
Vector3 ComplexLoop::ComputeBaryGradThetaFromOmegaPerface(int fid, const Vector3& bary) const {
  Vector3 gradTheta = Vector3::Zero();
  for (int i = 0; i < 3; i++) {
    gradTheta += bary[i] * ComputeGradThetaFromOmegaPerfaceCorner(fid, i);
  }
  return gradTheta;
}
} // namespace ComplexWrinkleField
