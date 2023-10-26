#include "CommonTools.h"
#include <Eigen/SPQRSupport>
#include <deque>
#include <filesystem>
#include <igl/cotmatrix_entries.h>
#include <igl/per_vertex_normals.h>
#include <iostream>
#include <queue>

Eigen::Vector3d ComputeHatWeight(Scalar u, Scalar v) {
  Eigen::Vector3d weights;
  Eigen::Vector3d bary(1 - u - v, u, v);
  for (int i = 0; i < 3; i++) {
    //		weights(i) = 3 * bary(i) * bary(i) - 2 * bary(i) * bary(i) * bary(i) + 2 * bary(i) * bary((i + 1) % 3) *
    // bary((i + 2) % 3);
    weights(i) = bary(i);
  }
  return weights;
}

MatrixX SPDProjection(MatrixX A) {
  MatrixX posHess = A;
  Eigen::SelfAdjointEigenSolver<MatrixX> es;
  es.compute(posHess);
  VectorX evals = es.eigenvalues();

  for (int i = 0; i < evals.size(); i++) {
    if (evals(i) < 0) evals(i) = 0;
  }
  MatrixX D = evals.asDiagonal();
  MatrixX V = es.eigenvectors();
  posHess = V * D * V.transpose();

  return posHess;
}

VectorX FaceVec2IntrinsicEdgeVec(const MatrixX& v, const Mesh& mesh) {
  int nedges = mesh.GetEdgeCount();
  VectorX edgeOmega(nedges);
  edgeOmega.setZero();

  for (int i = 0; i < nedges; i++) {
    int vid0 = mesh.GetEdgeVerts(i)[0];
    int vid1 = mesh.GetEdgeVerts(i)[1];

    Eigen::Vector3d e = mesh.GetVertPos(vid1) - mesh.GetVertPos(vid0);

    for (int j = 0; j < mesh.GetEdgeFaces(i).size(); j++) {
      int fid = mesh.GetEdgeFaces(i)[j];
      edgeOmega(i) += v.row(fid).dot(e) / mesh.GetEdgeFaces(i).size();
    }
  }
  return edgeOmega;
}

VectorX VertexVec2IntrinsicVec(const MatrixX& v, Mesh& mesh) {
  int nedges = mesh.GetEdgeCount();
  VectorX edgeOmega(nedges);

  for (int i = 0; i < nedges; i++) {
    int vid0 = mesh.GetEdgeVerts(i)[0];
    int vid1 = mesh.GetEdgeVerts(i)[1];

    Eigen::Vector3d e = mesh.GetVertPos(vid1) - mesh.GetVertPos(vid0);
    edgeOmega(i) = (v.row(vid0) + v.row(vid1)).dot(e) / 2;
  }
  return edgeOmega;
}

MatrixX IntrinsicEdgeVec2FaceVec(const VectorX& w, const Mesh& mesh) {
  int nfaces = mesh.GetFaceCount();

  MatrixX faceVec = MatrixX::Zero(nfaces, 3);
  for (int i = 0; i < nfaces; i++) {
    for (int j = 0; j < 3; j++) {
      int vid = mesh.GetFaceVerts(i)[j];

      int eid0 = mesh.GetFaceEdges(i)[j];
      int eid1 = mesh.GetFaceEdges(i)[(j + 2) % 3];

      Eigen::Vector3d e0 = mesh.GetVertPos(mesh.GetFaceVerts(i)[(j + 1) % 3]) - mesh.GetVertPos(vid);
      Eigen::Vector3d e1 = mesh.GetVertPos(mesh.GetFaceVerts(i)[(j + 2) % 3]) - mesh.GetVertPos(vid);

      int flag0 = 1, flag1 = 1;
      Eigen::Vector2d rhs;

      if (mesh.GetEdgeVerts(eid0)[0] == vid) {
        flag0 = 1;
      } else {
        flag0 = -1;
      }


      if (mesh.GetEdgeVerts(eid1)[0] == vid) {
        flag1 = 1;
      } else {
        flag1 = -1;
      }
      rhs(0) = flag0 * w(eid0);
      rhs(1) = flag1 * w(eid1);

      Eigen::Matrix2d I;
      I << e0.dot(e0), e0.dot(e1), e1.dot(e0), e1.dot(e1);
      Eigen::Vector2d sol = I.inverse() * rhs;

      faceVec.row(i) += (sol(0) * e0 + sol(1) * e1) / 3;
    }
  }
  return faceVec;
}

void Mkdir(const std::string& foldername) {
  if (!std::filesystem::exists(foldername)) {
    std::cout << "create directory: " << foldername << std::endl;
    if (!std::filesystem::create_directory(foldername)) {
      std::cerr << "create folder failed." << foldername << std::endl;
      exit(1);
    }
  }
}

VectorX GetFaceArea(const Mesh& mesh) {
  VectorX faceArea;
  MatrixX V;
  Eigen::MatrixXi F;
  mesh.GetPos(V);
  mesh.GetFace(F);

  igl::doublearea(V, F, faceArea);
  faceArea /= 2;
  return faceArea;
}

VectorX GetEdgeArea(const Mesh& mesh) {
  VectorX faceArea = GetFaceArea(mesh);
  VectorX edgeArea;
  edgeArea.setZero(mesh.GetEdgeCount());

  for (int i = 0; i < mesh.GetEdgeCount(); i++) {
    for (int j = 0; j < mesh.GetEdgeFaces(i).size(); j++) {
      int f0 = mesh.GetEdgeFaces(i)[j];
      edgeArea(i) += faceArea(f0) / mesh.GetEdgeFaces(i).size();
    }
  }
  return edgeArea;
}


VectorX GetVertArea(const Mesh& mesh) {
  VectorX faceArea = GetFaceArea(mesh);
  VectorX vertArea;
  vertArea.setZero(mesh.GetVertCount());

  for (int i = 0; i < mesh.GetFaceCount(); i++) {
    for (int j = 0; j < 3; j++) {
      int vid = mesh.GetFaceVerts(i)[j];
      vertArea(vid) += faceArea(i) / 3.;
    }
  }
  return vertArea;
}


void GetWrinkledMesh(const MatrixX& V, const Eigen::MatrixXi& F, const VectorX& zvals, MatrixX& wrinkledV,
                     Scalar scaleRatio, bool isTangentCorrection) {
  int nverts = V.rows();
  int nfaces = F.rows();

  wrinkledV = V;
  MatrixX VN;
  igl::per_vertex_normals(V, F, VN);

  for (int vid = 0; vid < nverts; vid++) {
    wrinkledV.row(vid) += scaleRatio * (zvals[vid] * VN.row(vid)); // only the real part is needed
  }

  if (isTangentCorrection) {
    std::vector<std::vector<Eigen::RowVector3d>> tanCorrections(nfaces);

    for (int i = 0; i < nfaces; i++) {
      for (int j = 0; j < 3; j++) {
        int vid = F(i, j);
        Eigen::Matrix2d Ib;
        Eigen::Matrix<Scalar, 3, 2> drb;
        drb.col(0) = (V.row(F(i, (j + 1) % 3)) - V.row(F(i, j))).transpose();
        drb.col(1) = (V.row(F(i, (j + 2) % 3)) - V.row(F(i, j))).transpose();

        Ib = drb.transpose() * drb;

        std::complex<Scalar> dz0 = zvals[F(i, (j + 1) % 3)] - zvals[F(i, j)];
        std::complex<Scalar> dz1 = zvals[F(i, (j + 2) % 3)] - zvals[F(i, j)];

        Eigen::Vector2d aSqdtheta;
        aSqdtheta << (std::conj(zvals[vid]) * dz0).imag(), (std::conj(zvals[vid]) * dz1).imag();

        Eigen::Vector3d extASqdtheta = drb * Ib.inverse() * aSqdtheta;

        Scalar theta = std::arg(zvals[vid]);
        Eigen::RowVector3d correction = scaleRatio * (1. / 8 * std::sin(2 * theta) * extASqdtheta.transpose());

        tanCorrections[vid].push_back(correction);
      }
    }

    for (int i = 0; i < nverts; i++)
      for (auto& c : tanCorrections[i]) wrinkledV.row(i) += c / tanCorrections[i].size();
  }
}


void ComputeBaryGradient(const Vector3& P0, const Vector3& P1, const Vector3& P2, const Vector3& bary, Matrix3& baryGrad) {
  // P = bary(0) * P0 + bary(1) * P1 + bary(2) * P2;
  Eigen::Matrix2d I, Iinv;

  I << (P1 - P0).squaredNorm(), (P1 - P0).dot(P2 - P0), (P2 - P0).dot(P1 - P0), (P2 - P0).squaredNorm();
  Iinv = I.inverse();

  Eigen::Matrix<Scalar, 3, 2> dr;
  dr.col(0) = P1 - P0;
  dr.col(1) = P2 - P0;

  Eigen::Matrix<Scalar, 2, 3> dbary12 = Iinv * dr.transpose();

  baryGrad.row(0) = -dbary12.row(0) - dbary12.row(1);
  baryGrad.row(1) = dbary12.row(0);
  baryGrad.row(2) = dbary12.row(1);
}



VectorX InconsistencyComputation(const Mesh& mesh, const VectorX& edgeW, const VectorX& zval) {
  int nverts = mesh.GetVertCount();
  int nfaces = mesh.GetFaceCount();
  VectorX incons = VectorX::Zero(nverts);
  for (int i = 0; i < nfaces; i++) {
    for (int j = 0; j < 3; j++) {
      int eid = mesh.GetFaceEdges(i)[j];
      int v0 = mesh.GetEdgeVerts(eid)[0];
      int v1 = mesh.GetEdgeVerts(eid)[1];

      Scalar theta0 = std::arg(std::complex<Scalar>(zval[v0], zval[v0 + nverts]));
      Scalar theta1 = std::arg(std::complex<Scalar>(zval[v1], zval[v1 + nverts]));

      incons(v0) += std::norm(std::complex<Scalar>(std::cos(theta0 + edgeW(eid)), std::sin(theta0 + edgeW(eid))) -
                              std::complex<Scalar>(std::cos(theta1), std::sin(theta0)));
      incons(v1) += std::norm(std::complex<Scalar>(std::cos(theta0 + edgeW(eid)), std::sin(theta0 + edgeW(eid))) -
                              std::complex<Scalar>(std::cos(theta1), std::sin(theta0)));
    }
  }
  return incons;
}

void RescaleZvals(const VectorX& normalizedZvals, const VectorX& norm, VectorX& zvals) {
  int size = normalizedZvals.size() / 2;
  if (!size || norm.rows() != size) return;
  zvals = normalizedZvals;

  for (int i = 0; i < size; i++) {
    zvals[i] = norm[i] * normalizedZvals[i];
    zvals[i + size] = norm[i] * normalizedZvals[i + size];
  }
}

VectorX NormalizeZvals(const VectorX& zvals) {
  int size = zvals.size() / 2;
  if (!size) return zvals;
  VectorX normalizedZvals = zvals;

  for (int i = 0; i < size; i++) {
    Scalar norm = std::sqrt(zvals[i] * zvals[i] + zvals[i + size] * zvals[i + size]);
    if (norm > 1e-10) {
      normalizedZvals[i] = zvals[i] / norm;
      normalizedZvals[i + size] = zvals[i + size] / norm;
    }
  }
  return normalizedZvals;
}

std::map<std::pair<int, int>, int> edgeMap(const std::vector<std::vector<int>>& edgeToVert) {
  std::map<std::pair<int, int>, int> heToEdge;
  for (int i = 0; i < edgeToVert.size(); i++) {
    std::pair<int, int> he = std::make_pair(edgeToVert[i][0], edgeToVert[i][1]);
    heToEdge[he] = i;
  }
  return heToEdge;
}

std::map<std::pair<int, int>, int> edgeMap(const Eigen::MatrixXi& faces) {
  std::map<std::pair<int, int>, int> edgeMap;
  std::vector<std::vector<int>> edgeToVert;
  for (int face = 0; face < faces.rows(); ++face) {
    for (int i = 0; i < 3; ++i) {
      int vi = faces(face, i);
      int vj = faces(face, (i + 1) % 3);
      assert(vi != vj);

      std::pair<int, int> he = std::make_pair(vi, vj);
      if (he.first > he.second) std::swap(he.first, he.second);
      if (edgeMap.find(he) != edgeMap.end()) continue;

      edgeMap[he] = edgeToVert.size();
      edgeToVert.push_back(std::vector<int>(2));
      edgeToVert.back()[0] = he.first;
      edgeToVert.back()[1] = he.second;
    }
  }
  return edgeMap;
}

VectorX SwapEdgeVec(const std::vector<std::vector<int>>& edgeToVert, const VectorX& edgeVec, int isInputConsistent) {
  VectorX edgeVecSwap = edgeVec;
  std::map<std::pair<int, int>, int> heToEdge = edgeMap(edgeToVert);

  int idx = 0;
  for (auto it : heToEdge) {
    if (isInputConsistent == 0)
      edgeVecSwap(it.second) = edgeVec(idx);
    else
      edgeVecSwap(idx) = edgeVec(it.second);
    idx++;
  }
  return edgeVecSwap;
}

VectorX SwapEdgeVec(const Eigen::MatrixXi& faces, const VectorX& edgeVec, int isInputConsistent) {
  VectorX edgeVecSwap = edgeVec;
  std::map<std::pair<int, int>, int> heToEdge = edgeMap(faces);

  int idx = 0;
  for (auto it : heToEdge) {
    if (isInputConsistent == 0)
      edgeVecSwap(it.second) = edgeVec(idx);
    else
      edgeVecSwap(idx) = edgeVec(it.second);
    idx++;
  }
  return edgeVecSwap;
}