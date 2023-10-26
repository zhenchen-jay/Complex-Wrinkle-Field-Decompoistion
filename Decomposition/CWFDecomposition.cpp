#include "CWFDecomposition.h"

#include <igl/adjacency_list.h>
#include <igl/hausdorff.h>
#include <igl/per_vertex_normals.h>
#include <igl/writeOBJ.h>

#include "../KnoppelStripePatterns.h"
#include "../LoadSaveIO.h"
#include "../MeshLib/IntrinsicGeometry.h"
#include "../Optimization/Newton.h"
#include "../Optimization/ProjectedNewton.h"
#include "../Optimization/TestGradHess.h"
#include "../Upsampling/Subdivision.h"

namespace ComplexWrinkleField {
void CWFDecomposition::initialization(const CWF& cwf, int upsampleTimes) // should be removed!
{
  _baseCWF = cwf;
  _upsampleTimes = upsampleTimes;
  CWF upcwf;
  Subdivide(cwf, upcwf, upsampleTimes, false, &_LoopS0, nullptr, &_upZMat);
  _upMesh = upcwf._mesh;
  _upMesh.GetPos(_upV);
  _upMesh.GetFace(_upF);
  igl::per_vertex_normals(_upV, _upF, _upN);
  _baseEdgeArea = getEdgeArea(_baseCWF._mesh);
  _baseVertArea = getVertArea(_baseCWF._mesh);
  _upVertArea = getVertArea(_upMesh);

  updateWrinkleCompUpMat();
}

void CWFDecomposition::initialization(const CWF& cwf, int upsampleTimes, // CWF info
                                      const Mesh& restMesh,              // rest (coarse) mesh
                                      const Mesh& restWrinkleMesh,       // rest (wrinkle) mesh
                                      const Mesh& wrinkledMesh,          // target wrinkle mesh (for decomposition)
                                      double youngsModulus,              // Young's Modulus
                                      double poissonRatio,               // Poisson's Ratio
                                      double thickness                   // thickness
) {
  _baseCWF = cwf;
  _upsampleTimes = upsampleTimes;
  CWF upcwf;
  Subdivide(cwf, upcwf, upsampleTimes, false, &_LoopS0, nullptr, &_upZMat);


  _upMesh = upcwf._mesh;
  _upMesh.GetPos(_upV);
  _upMesh.GetFace(_upF);
  igl::per_vertex_normals(_upV, _upF, _upN);
  _baseEdgeArea = getEdgeArea(_baseCWF._mesh);
  _baseVertArea = getVertArea(_baseCWF._mesh);
  _upVertArea = getVertArea(_upMesh);

  _restMesh = restMesh;
  _restWrinkledMesh = restWrinkleMesh;
  _wrinkledMesh = wrinkledMesh;
  _wrinkledMesh.GetPos(_wrinkledV);
  _wrinkledMesh.GetFace(_wrinkledF);

  // this is really annoying. Some how we need to unify the mesh connectivity!
  Eigen::MatrixXd restPos, curPos;
  Eigen::MatrixXi restF, curF;
  _restMesh.GetPos(restPos);
  _restMesh.GetFace(restF);

  _baseCWF._mesh.GetPos(curPos);
  _baseCWF._mesh.GetFace(curF);

  MeshConnectivity restMeshCon(restF), curMeshCon(curF);

  // clamped amp and omega
  buildProjectionMat({}, curMeshCon, curPos.rows());
  tfwShell =
      WrinkledTensionField::TFWShell(restPos, restMeshCon, curPos, curMeshCon, poissonRatio, thickness, youngsModulus);
  tfwShell.initialization();

  updateWrinkleCompUpMat();
}


void CWFDecomposition::initialization(int upsampleTimes,           // upsample info
                                      bool isFixedBnd,             // fixe bnd for loop
                                      const Mesh& restMesh,        // rest (coarse) mesh
                                      const Mesh& baseMesh,        // base (coarse) mesh
                                      const Mesh& restWrinkleMesh, // rest (wrinkle) mesh
                                      const Mesh& wrinkledMesh,    // target wrinkle mesh (for decomposition)
                                      double youngsModulus,        // Young's Modulus
                                      double poissonRatio,         // Poisson's Ratio
                                      double thickness,            // thickness
                                      const std::unordered_set<int>& clampedVert // clamped vertices
) {
  _upsampleTimes = upsampleTimes;
  _restMesh = restMesh;
  _restWrinkledMesh = restWrinkleMesh;
  _wrinkledMesh = wrinkledMesh;
  _wrinkledMesh.GetPos(_wrinkledV);
  _wrinkledMesh.GetFace(_wrinkledF);


  // this is really annoying. Some how we need to unify the mesh connectivity!
  Eigen::MatrixXd restPos, curPos;
  Eigen::MatrixXi restF, curF;
  _restMesh.GetPos(restPos);
  _restMesh.GetFace(restF);

  baseMesh.GetPos(curPos);
  baseMesh.GetFace(curF);

  MeshConnectivity restMeshCon(restF), curMeshCon(curF);
  // clamped amp and omega
  buildProjectionMat(clampedVert, curMeshCon, curPos.rows());
  tfwShell =
      WrinkledTensionField::TFWShell(restPos, restMeshCon, curPos, curMeshCon, poissonRatio, thickness, youngsModulus);
  tfwShell.initialization();

  _baseEdgeArea = getEdgeArea(baseMesh);
  _baseVertArea = getVertArea(baseMesh);

  // initialize amp and omega according to the compression
  double d1, d2;
  igl::hausdorff(_restWrinkledMesh.GetPos(), _restWrinkledMesh.GetFace(), _wrinkledV, _wrinkledF, d1);
  igl::hausdorff(curPos, curF, _wrinkledV, _wrinkledF, d2);
  double ampGuess = std::min(d1, d2);
  if (ampGuess < 1e-3) // too small
  {
    ampGuess = 0.1;
    double length = curPos.row(0).maxCoeff() - curPos.row(0).minCoeff();
    double width = curPos.row(1).maxCoeff() - curPos.row(1).minCoeff();
    double height = curPos.row(2).maxCoeff() - curPos.row(2).minCoeff();

    double bboxSize = std::max(length, width);
    bboxSize = std::max(bboxSize, height);

    ampGuess = 0.1 * bboxSize;
  }
  Eigen::VectorXd amp, omega;
  Eigen::MatrixXd faceOmega;
  initializeAmpOmega(curPos, curMeshCon, ampGuess, amp, faceOmega);
  omega = faceVec2IntrinsicEdgeVec(faceOmega, baseMesh);
  omega *= 17.0 / 13.0;
  amp /= 17.0 / 13.0;

  // initialize zvals
  VectorX zvals;
  roundZvalsFromEdgeOmega(baseMesh, omega, _baseEdgeArea, _baseVertArea, baseMesh.GetVertCount(), zvals);
  _baseCWF = CWF(amp, omega, zvals, baseMesh);
  CWF upcwf;
  Subdivide(_baseCWF, upcwf, upsampleTimes, false, &_LoopS0, nullptr, &_upZMat);
  _upMesh = upcwf._mesh;
  _upMesh.GetPos(_upV);
  _upMesh.GetFace(_upF);
  igl::per_vertex_normals(_upV, _upF, _upN);
  _upVertArea = getVertArea(_upMesh);

  updateWrinkleCompUpMat();
}

void CWFDecomposition::initialization(const CWF& cwf, int upsampleTimes,
                                      bool isFixedBnd,             // fixed bnd for loop
                                      const Mesh& restMesh,        // rest (coarse) mesh
                                      const Mesh& restWrinkleMesh, // rest (wrinkle) mesh
                                      const Mesh& wrinkledMesh,    // target wrinkle mesh (for decomposition)
                                      double youngsModulus,        // Young's Modulus
                                      double poissonRatio,         // Poisson's Ratio
                                      double thickness,            // thickness
                                      const std::unordered_set<int>& clampedVert // clamped vertices
) {
  _upsampleTimes = upsampleTimes;
  _restMesh = restMesh;
  _restWrinkledMesh = restWrinkleMesh;
  _wrinkledMesh = wrinkledMesh;
  _wrinkledMesh.GetPos(_wrinkledV);
  _wrinkledMesh.GetFace(_wrinkledF);


  // this is really annoying. Some how we need to unify the mesh connectivity!
  Eigen::MatrixXd restPos, curPos;
  Eigen::MatrixXi restF, curF;
  _restMesh.GetPos(restPos);
  _restMesh.GetFace(restF);

  cwf._mesh.GetPos(curPos);
  cwf._mesh.GetFace(curF);

  MeshConnectivity restMeshCon(restF), curMeshCon(curF);
  // clamped amp and omega
  buildProjectionMat(clampedVert, curMeshCon, curPos.rows());
  tfwShell =
      WrinkledTensionField::TFWShell(restPos, restMeshCon, curPos, curMeshCon, poissonRatio, thickness, youngsModulus);
  tfwShell.initialization();

  _baseEdgeArea = getEdgeArea(cwf._mesh);
  _baseVertArea = getVertArea(cwf._mesh);
  _baseCWF = cwf;

  CWF upcwf;
  Subdivide(cwf, upcwf, upsampleTimes, false, &_LoopS0, nullptr, &_upZMat);
  _upMesh = upcwf._mesh;
  _upMesh.GetPos(_upV);
  _upMesh.GetFace(_upF);
  igl::per_vertex_normals(_upV, _upF, _upN);
  _upVertArea = getVertArea(_upMesh);

  updateWrinkleCompUpMat();
}

struct UnionFind {
  std::vector<int> parent;
  std::vector<int> sign;
  UnionFind(int items) {
    parent.resize(items);
    sign.resize(items);
    for (int i = 0; i < items; i++) {
      parent[i] = i;
      sign[i] = 1;
    }
  }

  std::pair<int, int> find(int i) {
    if (parent[i] != i) {
      auto newparent = find(parent[i]);
      sign[i] *= newparent.second;
      parent[i] = newparent.first;
    }

    return {parent[i], sign[i]};
  }

  void dounion(int i, int j, int usign) {
    auto xroot = find(i);
    auto yroot = find(j);
    if (xroot.first != yroot.first) {
      parent[xroot.first] = yroot.first;
      sign[xroot.first] = usign * xroot.second * yroot.second;
    }
  }
};

static const double PI = 3.1415926535898;

static void combField(const Eigen::MatrixXi& F, const std::vector<Eigen::Matrix2d>& abars,
                      const Eigen::VectorXd* weight, const Eigen::MatrixXd& w, Eigen::MatrixXd& combedW) {
  int nfaces = F.rows();
  UnionFind uf(nfaces);
  MeshConnectivity mesh(F);
  IntrinsicGeometry geom(mesh, abars);
  struct Visit {
    int edge;
    int sign;
    double norm;
    bool operator<(const Visit& other) const { return norm > other.norm; }
  };

  std::priority_queue<Visit> pq;

  int nedges = mesh.nEdges();
  for (int i = 0; i < nedges; i++) {
    int face1 = mesh.edgeFace(i, 0);
    int face2 = mesh.edgeFace(i, 1);
    if (face1 == -1 || face2 == -1) continue;

    Eigen::Vector2d curw = abars[face1].inverse() * w.row(face1).transpose();
    Eigen::Vector2d nextwbary1 = abars[face2].inverse() * w.row(face2).transpose();
    Eigen::Vector2d nextw = geom.Ts.block<2, 2>(2 * i, 2) * nextwbary1;
    int sign = ((curw.transpose() * abars[face1] * nextw) < 0 ? -1 : 1);
    double innerp = curw.transpose() * abars[face1] * nextw;

    if (!weight) {
      double normcw = std::sqrt(curw.transpose() * abars[face1] * curw);
      double normnw = std::sqrt(nextw.transpose() * abars[face1] * nextw);
      double negnorm = -std::min(normcw, normnw);
      pq.push({i, sign, negnorm});

    } else {
      double compcw = weight->coeffRef(face1);
      double compnw = weight->coeffRef(face2);
      double negw = -std::min(compcw, compnw);

      pq.push({i, sign, negw});
    }
  }

  while (!pq.empty()) {
    auto next = pq.top();
    pq.pop();
    uf.dounion(mesh.edgeFace(next.edge, 0), mesh.edgeFace(next.edge, 1), next.sign);
  }

  combedW.resize(nfaces, 2);
  for (int i = 0; i < nfaces; i++) {
    int sign = uf.find(i).second;
    combedW.row(i) = w.row(i) * sign;
  }
}

void CWFDecomposition::initializeAmpOmega(const Eigen::MatrixXd& curPos, MeshConnectivity& curMeshCon, double ampGuess,
                                          Eigen::VectorXd& amp, Eigen::MatrixXd& faceOmega) {
  // compute the compression direction per face
  int nfaces = curMeshCon.nFaces();
  int nverts = curPos.rows();
  amp.resize(nverts);
  amp.setConstant(ampGuess);

  Eigen::MatrixXd faceOmegaPara(nfaces, 2);

  for (int i = 0; i < nfaces; i++) {
    Eigen::Matrix2d abar = tfwShell.getIbar(i);
    Eigen::Matrix2d a = tfwShell.getI(i);
    Eigen::Matrix2d diff = a - abar;
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix2d> solver(diff, abar);
    Eigen::Vector2d evec = solver.eigenvectors().col(0);

    double faceCompression = fabs(solver.eigenvalues()[0]);
    faceOmegaPara.row(i) =
        std::sqrt(2.0 * faceCompression) * (abar * evec).transpose(); // convert to one-form, assume unit amp
  }

  // comb vectors
  Eigen::MatrixXd combedOmega;
  combField(curMeshCon.faces(), tfwShell.getIbars(), nullptr, faceOmegaPara, combedOmega);
  std::swap(faceOmegaPara, combedOmega);
  faceOmegaPara /= ampGuess;

  faceOmega.resize(nfaces, 3);
  for (int i = 0; i < nfaces; i++) {
    Eigen::Matrix2d abar = tfwShell.getIbar(i);
    // get covector
    Eigen::Vector2d covector = abar.inverse() * (faceOmegaPara.row(i).transpose());

    // get the vector in 3D
    Eigen::RowVector3d ru = curPos.row(curMeshCon.faceVertex(i, 1)) - curPos.row(curMeshCon.faceVertex(i, 0));
    Eigen::RowVector3d rv = curPos.row(curMeshCon.faceVertex(i, 2)) - curPos.row(curMeshCon.faceVertex(i, 0));
    faceOmega.row(i) = covector[0] * ru + covector[1] * rv;
  }

  // clamped vertice indicates that amplitude is zero
  for (auto& it : _clampedVertices) amp[it] = 0;

  // smoothing for the amplitude
  std::vector<std::vector<int>> vertNeis;
  igl::adjacency_list(curMeshCon.faces(), vertNeis);

  Eigen::SparseMatrix<double> L(curPos.rows(), curPos.rows());
  std::vector<Eigen::Triplet<double>> Lcoeff;

  for (int i = 0; i < curPos.rows(); i++) {
    if (amp(i) > 0) // not pure tension vertices or clamped vertices
    {
      for (auto& neiV : vertNeis[i]) {
        Lcoeff.push_back(Eigen::Triplet<double>(i, neiV, 1.0 / vertNeis[i].size()));
      }
    } else
      Lcoeff.push_back(Eigen::Triplet<double>(i, i, 1.0));
  }

  L.setFromTriplets(Lcoeff.begin(), Lcoeff.end());

  // apply twice by default
  amp = L * (L * amp);
}

void CWFDecomposition::updateWrinkleCompUpMat() {
  CWF upcwf;
  Subdivide(_baseCWF, upcwf, _upsampleTimes, false, nullptr, nullptr, &_upZMat);
  std::vector<TripletX> T;
  assert(_upZMat.rows() == 2 * upcwf._mesh.GetVertCount());
  for (int k = 0; k < _upZMat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<Scalar>::InnerIterator it(_upZMat, k); it; ++it) {
      if (it.row() < upcwf._mesh.GetVertCount()) {
        T.push_back({TripletX(it.row(), it.col(), it.value())});
      }
    }
  }

  _wrinkleCompUpMat.resize(_upZMat.rows() / 2, _upZMat.cols());
  _wrinkleCompUpMat.setFromTriplets(T.begin(), T.end());
}

void CWFDecomposition::buildProjectionMat(const std::unordered_set<int>& clampedVerts, const MeshConnectivity& meshCon,
                                          int nverts) {
  _nFreeAmp = nverts;
  int nedges = meshCon.nEdges();
  _nFreeOmega = nedges;
  _clampedVertices = clampedVerts;

  _clampedAmpOmega.clear();
  for (auto& it : clampedVerts) {
    _clampedAmpOmega.insert(it);
    _nFreeAmp--;
  }

  for (int e = 0; e < nedges; e++) {
    int vid0 = meshCon.edgeVertex(e, 0);
    int vid1 = meshCon.edgeVertex(e, 1);

    if (clampedVerts.count(vid0) && clampedVerts.count(vid1)) {
      _clampedAmpOmega.insert(e + nverts);
      _nFreeOmega--;
    }
  }

  int freeDOFs = _nFreeAmp + _nFreeOmega;
  int row = 0;

  std::vector<TripletX> projT, projposT;
  for (int i = 0; i < nedges + nverts; i++) {
    if (_clampedAmpOmega.count(i)) continue;
    projT.push_back(Eigen::Triplet<double>(row, i, 1.0));
    row++;
  }
  _projTFWMat.resize(freeDOFs, nedges + nverts);
  _projTFWMat.setFromTriplets(projT.begin(), projT.end());

  _unprojTFWMat = _projTFWMat.transpose();


  std::cout << "Free amps: " << _nFreeAmp << std::endl;
  std::cout << "Free omegas: " << _nFreeOmega << std::endl;

  // projection matrix for base mesh positions
  freeDOFs = 3 * _nFreeAmp;
  row = 0;
  for (int i = 0; i < nverts; i++) {
    if (clampedVerts.count(i)) continue;

    for (int j = 0; j < 3; j++) {
      projposT.push_back(Eigen::Triplet<double>(row, 3 * i + j, 1.0));
      row++;
    }
  }

  _projPosMat.resize(freeDOFs, 3 * nverts);
  _projPosMat.setFromTriplets(projposT.begin(), projposT.end());

  _unprojPosMat = _projPosMat.transpose();
}

void CWFDecomposition::getCWF(CWF& cwf) { cwf = _baseCWF; }

void CWFDecomposition::optimizeAmpOmega() {
  tfwShell.updateBaseGeometries(_baseCWF._mesh.GetPos());
  Eigen::VectorXd amp = _baseCWF._amp;
  Eigen::VectorXd omega = _baseCWF._omega;

  int nverts = _baseCWF._mesh.GetVertCount();
  int nedges = _baseCWF._mesh.GetEdgeCount();

  omega = swapEdgeVec(_baseCWF._mesh.GetFace(), omega, 1);
  auto omega0 = omega;
  auto amp0 = amp;

  Eigen::VectorXd fixedFullx(nverts + nedges);
  for (int i = 0; i < nverts + nedges; i++) {
    if (_clampedAmpOmega.count(i)) {
      fixedFullx[i] = i < nverts ? amp[i] : omega[i - nverts];
    } else {
      fixedFullx[i] = 0;
    }
  }

  auto convertFromTFW = [&](Eigen::VectorXd& x) {
    x.resize(nverts + nedges);

    for (int i = 0; i < nverts; i++) {
      x[i] = amp[i];
    }
    for (int i = 0; i < nedges; i++) {
      x[nverts + i] = omega[i];
    }

    x = _projTFWMat * x;
  };

  auto convert2TFW = [&](const Eigen::VectorXd& x) {
    Eigen::VectorXd fullx = _unprojTFWMat * x + fixedFullx;
    for (int i = 0; i < nverts; i++) {
      amp[i] = fullx[i];
    }
    for (int i = 0; i < nedges; i++) {
      omega[i] = fullx[nverts + i];
    }
  };

  auto TFWEnergy = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess,
                       bool isProj) {
    convert2TFW(x); // convert the TFW variables:  amp and omega. Attention: TFW.omega is a permutation of CWF.omega
    double energy = tfwShell.elasticReducedEnergy(amp, omega, grad, hess, isProj);

    if (grad) (*grad) = _projTFWMat * (*grad);
    if (hess) (*hess) = _projTFWMat * (*hess) * _unprojTFWMat;

    return energy;
  };

  auto findMaxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) -> double { return 1.0; };

  Eigen::VectorXd x, lx, ux;
  convertFromTFW(x);
  lx = x;
  ux = x;

  ux.setConstant(std::numeric_limits<double>::max());
  lx.segment(0, _nFreeAmp).setZero();
  lx.segment(_nFreeAmp, _nFreeOmega).setConstant(std::numeric_limits<double>::min());

  // projectedNewtonSolver(TFWEnergy, findMaxStep, lx, ux, x, 10, 1e-6, 1e-15, 1e-15, true);
  NewtonSolver(TFWEnergy, findMaxStep, x, 10, 1e-6, 1e-15, 1e-15, true);

  convert2TFW(x);

  std::cout << "amp update: " << (amp - amp0).norm() << std::endl;
  std::cout << "omega update: " << (omega - omega0).norm() << std::endl;


  _baseCWF._amp = amp;
  _baseCWF._omega = swapEdgeVec(_baseCWF._mesh.GetFace(), omega, 0);
}

void CWFDecomposition::optimizePhase() {
  std::cout << "\n update vertex phase: " << std::endl;
  precomputationForPhase();
  VectorX zvec = _baseCWF._zvals;
  int nverts = _baseCWF._mesh.GetVertCount();

  VectorX z0 = zvec;
  z0.setZero();
  double diff0 = computeDifferenceFromZvals(z0);

  auto zvalEnergy = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess,
                        bool isProj) {
    double compatEnergy = 0.5 * x.dot(_zvalCompHess * x);
    double diffEnergy = 0.5 * x.dot(_zvalDiffHess * x) + _zvalDiffCoeff.dot(x) + diff0;

    Eigen::VectorXd unitGrad;
    SparseMatrixX unitHess;
    double unitEnergy = computeUnitNormEnergy(zvec, grad ? &unitGrad : nullptr, hess ? &unitHess : nullptr, isProj);

    /*double compatEnergy = computeCompatibilityEnergy(_baseCWF._omega, zvec);
    double diffEnergy = computeDifferenceFromZvals(zvec);*/

    double rc = 1e-3;
    double runit = 1e-3;

    if (grad || hess) {
      if (grad) {
        (*grad) = (_zvalDiffHess * x + _zvalDiffCoeff) + rc * _zvalCompHess * x + runit * unitGrad;
      }
      if (hess) (*hess) = _zvalDiffHess + rc * _zvalCompHess + runit * unitHess;
    }
    return diffEnergy + rc * compatEnergy + runit * unitEnergy;
  };

  auto findMaxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) -> double { return 1.0; };

  NewtonSolver(zvalEnergy, findMaxStep, zvec, 1000, 1e-6, 1e-15, 1e-15, true);
  _baseCWF._zvals = zvec;
}

void CWFDecomposition::optimizeBasemesh() {
  std::cout << "\n update base mesh" << std::endl;
  // base mesh update
  precomptationForBaseMesh();
  MatrixX pos = _baseCWF._mesh.GetPos();
  int nverts = _baseCWF._mesh.GetVertCount();

  MatrixX zeros = pos;
  zeros.setZero();
  double diff0 = computeDifferenceFromBasemesh(zeros);

  VectorX fixedPos = VectorX::Zero(3 * nverts);
  std::vector<Eigen::Triplet<double>> freeT;
  int nrows = 0;
  for (auto& it : _clampedVertices) {
    for (int j = 0; j < 3; j++) {
      fixedPos[3 * it + j] = pos(it, j);
    }
  }
  for (int i = 0; i < nverts; i++) {
    if (_clampedVertices.count(i)) {
      continue;
    } else {
      for (int j = 0; j < 3; j++) {
        freeT.push_back(Eigen::Triplet<double>(nrows++, 3 * i + j, pos(i, j)));
      }
    }
  }
  _projPosMat.resize(nrows, 3 * nverts);
  _projPosMat.setFromTriplets(freeT.begin(), freeT.end());
  _unprojPosMat = _projPosMat.transpose();

  auto posEnergy = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess,
                       bool isProj) {
    Eigen::VectorXd fullx = _unprojPosMat * x + fixedPos;
    double energy = 0.5 * fullx.dot(_baseMeshDiffHess * fullx) + fullx.dot(_baseMeshDiffCoeff) + diff0;
    if (grad) {
      (*grad) = _baseMeshDiffHess * fullx + _baseMeshDiffCoeff;
      (*grad) = _projPosMat * (*grad);
    }

    if (hess) {
      *hess = _baseMeshDiffHess;
      (*hess) = _projPosMat * (*hess) * _unprojPosMat;
    }

    return energy;
  };

  auto findMaxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) -> double { return 1.0; };

  Eigen::VectorXd x = _projPosMat * flatMatrix(pos);
  NewtonSolver(posEnergy, findMaxStep, x, 1000, 1e-6, 1e-15, 1e-15, true);
  Eigen::VectorXd fullx = _unprojPosMat * x + fixedPos;

  /*Eigen::MatrixXd newPos = unFlatVector(fullx, 3);
  for (int i = 0; i < newPos.rows(); i++)
  {
      std::cout << newPos.row(i) << " || " << _baseCWF._mesh.GetVertPos(i).transpose() << std::endl;
  }*/
  igl::writeOBJ("after_update_basemesh.obj", unFlatVector(fullx, 3), _baseCWF._mesh.GetFace());
  _baseCWF._mesh.SetPos(unFlatVector(fullx, 3));
}

void CWFDecomposition::optimizeCWF() {

  // alternative update
  for (int i = 0; i < 1; i++) {
    // optimizeAmpOmega();		// TFW sucks
    //		Eigen::VectorXd amp, omega;
    //		Eigen::MatrixXd faceOmega;
    //
    //		Eigen::MatrixXd curPos;
    //		Eigen::MatrixXi curF;
    //
    //		_baseCWF._mesh.GetPos(curPos);
    //		_baseCWF._mesh.GetFace(curF);
    //
    //		MeshConnectivity curMeshCon(curF);
    //
    //		initializeAmpOmega(curPos, curMeshCon, _baseCWF._amp.maxCoeff(), amp, faceOmega);
    //		omega = faceVec2IntrinsicEdgeVec(faceOmega, _baseCWF._mesh);
    //		_baseCWF._amp = amp;
    //		_baseCWF._omega = omega;

    optimizePhase();
    optimizeBasemesh();
  }
}

///////////////////////////////////  vertex phase update energies
/////////////////////////////////////////////////////////////
void CWFDecomposition::precomputationForPhase() {
  updateWrinkleCompUpMat();
  _upAmp = _LoopS0 * _baseCWF._amp;

  // compaptibility hessian
  std::vector<TripletX> AT;
  int nedges = _baseCWF._mesh.GetEdgeCount();
  int nverts = _baseCWF._mesh.GetVertCount();

  for (int i = 0; i < nedges; i++) {
    int vid0 = _baseCWF._mesh.GetEdgeVerts(i)[0];
    int vid1 = _baseCWF._mesh.GetEdgeVerts(i)[1];


    std::complex<double> expw0 = std::complex<double>(std::cos(_baseCWF._omega(i)), std::sin(_baseCWF._omega(i)));

    double ce = _baseEdgeArea[i];

    AT.push_back({vid0, vid0, ce});
    AT.push_back({vid0 + nverts, vid0 + nverts, ce});

    AT.push_back({vid1, vid1, ce});
    AT.push_back({vid1 + nverts, vid1 + nverts, ce});

    AT.push_back({vid0, vid1, -ce * (expw0.real())});
    AT.push_back({vid0 + nverts, vid1, -ce * (-expw0.imag())});
    AT.push_back({vid0, vid1 + nverts, -ce * (expw0.imag())});
    AT.push_back({vid0 + nverts, vid1 + nverts, -ce * (expw0.real())});

    AT.push_back({vid1, vid0, -ce * (expw0.real())});
    AT.push_back({vid1, vid0 + nverts, -ce * (-expw0.imag())});
    AT.push_back({vid1 + nverts, vid0, -ce * (expw0.imag())});
    AT.push_back({vid1 + nverts, vid0 + nverts, -ce * (expw0.real())});
  }

  _zvalCompHess.resize(2 * nverts, 2 * nverts);
  _zvalCompHess.setFromTriplets(AT.begin(), AT.end());

  // zval difference hess
  int nupverts = _upV.rows();

  _zvalDiffCoeff.resize(nupverts);
  std::vector<TripletX> T;

  for (int i = 0; i < nupverts; i++) {
    T.push_back({i, i, _upAmp[i] * _upAmp[i] * _upVertArea[i]});
    _zvalDiffCoeff[i] = -_upAmp[i] * _upVertArea[i] * (_wrinkledV.row(i) - _upV.row(i)).dot(_upN.row(i));
  }

  SparseMatrixX diag(nupverts, nupverts), _zvalCompHess(2 * nverts, 2 * nverts);
  diag.setFromTriplets(T.begin(), T.end());

  _zvalDiffHess = _wrinkleCompUpMat.transpose() * diag * _wrinkleCompUpMat;
  _zvalDiffCoeff = _wrinkleCompUpMat.transpose() * _zvalDiffCoeff;
}

double CWFDecomposition::computeCompatibilityEnergy(const VectorX& omega, const VectorX& zvals, VectorX* grad,
                                                    SparseMatrixX* hess) {
  std::vector<TripletX> AT;
  int nedges = _baseCWF._mesh.GetEdgeCount();
  int nverts = _baseCWF._mesh.GetVertCount();
  double energy = 0;

  for (int i = 0; i < nedges; i++) {
    int vid0 = _baseCWF._mesh.GetEdgeVerts(i)[0];
    int vid1 = _baseCWF._mesh.GetEdgeVerts(i)[1];

    std::complex<double> z0(zvals[vid0], zvals[vid0 + nverts]);
    std::complex<double> z1(zvals[vid1], zvals[vid1 + nverts]);

    std::complex<double> expw0 = std::complex<double>(std::cos(omega(i)), std::sin(omega(i)));

    double ce = _baseEdgeArea[i];
    energy += 0.5 * norm((z0 * expw0 - z1)) * ce;


    if (grad || hess) {
      AT.push_back({vid0, vid0, ce});
      AT.push_back({vid0 + nverts, vid0 + nverts, ce});

      AT.push_back({vid1, vid1, ce});
      AT.push_back({vid1 + nverts, vid1 + nverts, ce});

      AT.push_back({vid0, vid1, -ce * (expw0.real())});
      AT.push_back({vid0 + nverts, vid1, -ce * (-expw0.imag())});
      AT.push_back({vid0, vid1 + nverts, -ce * (expw0.imag())});
      AT.push_back({vid0 + nverts, vid1 + nverts, -ce * (expw0.real())});

      AT.push_back({vid1, vid0, -ce * (expw0.real())});
      AT.push_back({vid1, vid0 + nverts, -ce * (-expw0.imag())});
      AT.push_back({vid1 + nverts, vid0, -ce * (expw0.imag())});
      AT.push_back({vid1 + nverts, vid0 + nverts, -ce * (expw0.real())});
    }
  }

  if (grad || hess) {
    Eigen::SparseMatrix<double> A;

    A.resize(2 * nverts, 2 * nverts);
    A.setFromTriplets(AT.begin(), AT.end());

    if (grad) {
      (*grad) = A * zvals;
    }

    if (hess) {
      hess->resize(2 * nverts, 2 * nverts);
      hess->setFromTriplets(AT.begin(), AT.end());
    }
  }
  return energy;
}

double CWFDecomposition::computeDifferenceFromZvals(const VectorX& zvals, VectorX* grad, SparseMatrixX* hess) {
  VectorX upZvals = _upZMat * zvals;

  double energy = 0;
  int nupverts = _upV.rows();
  int nverts = zvals.rows() / 2;

  VectorX coeff;
  if (grad || hess) coeff.resize(nupverts);
  std::vector<TripletX> T;

  for (int i = 0; i < nupverts; i++) {
    double diff =
        0.5 * (_wrinkledV.row(i) - _upV.row(i) - _upAmp[i] * upZvals[i] * _upN.row(i)).squaredNorm() * _upVertArea[i];

    energy += diff;

    if (grad || hess) {
      T.push_back({i, i, _upAmp[i] * _upAmp[i] * _upVertArea[i]});
      coeff[i] = -_upAmp[i] * _upVertArea[i] * (_wrinkledV.row(i) - _upV.row(i)).dot(_upN.row(i));
    }
  }

  if (grad || hess) {
    SparseMatrixX diag(nupverts, nupverts), H(2 * nverts, 2 * nverts);
    diag.setFromTriplets(T.begin(), T.end());

    H = _wrinkleCompUpMat.transpose() * diag * _wrinkleCompUpMat;

    if (grad) {
      (*grad) = H * zvals + _wrinkleCompUpMat.transpose() * coeff;
    }

    if (hess) (*hess) = H;
  }

  return energy;
}

double CWFDecomposition::computeUnitNormEnergy(const VectorX& zvals, VectorX* grad, SparseMatrixX* hess, bool isProj) {
  int nverts = zvals.rows() / 2;
  double energy = 0;

  if (grad) grad->setZero(2 * nverts);

  std::vector<TripletX> T;

  for (int vid = 0; vid < nverts; vid++) {
    double ampSq = zvals[vid] * zvals[vid] + zvals[vid + nverts] * zvals[vid + nverts];
    double refAmpSq = 1;

    double ca = _baseVertArea[vid];

    energy += ca * (ampSq - refAmpSq) * (ampSq - refAmpSq);

    if (grad) {
      (*grad)(vid) += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * zvals[vid]);
      (*grad)(vid + nverts) += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * zvals[vid + nverts]);
    }

    if (hess) {
      Eigen::Matrix2d tmpHess;
      tmpHess << 2.0 * zvals[vid] * 2.0 * zvals[vid], 2.0 * zvals[vid] * 2.0 * zvals[vid + nverts],
          2.0 * zvals[vid] * 2.0 * zvals[vid + nverts], 2.0 * zvals[vid + nverts] * 2.0 * zvals[vid + nverts];

      tmpHess *= 2.0 * ca;
      tmpHess += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * Eigen::Matrix2d::Identity());


      if (isProj) tmpHess = SPDProjection(tmpHess);

      for (int k = 0; k < 2; k++)
        for (int l = 0; l < 2; l++) T.push_back({vid + k * nverts, vid + l * nverts, tmpHess(k, l)});
    }
  }

  if (hess) {
    hess->resize(2 * nverts, 2 * nverts);
    hess->setFromTriplets(T.begin(), T.end());
  }
  return energy;
}

void CWFDecomposition::testDifferenceFromZvals(const VectorX& zvals) {
  VectorX z0 = zvals, z = zvals;
  z0.setZero();
  double f0 = computeDifferenceFromZvals(z0);

  z.setRandom();
  double f = computeDifferenceFromZvals(z);
  int nverts = z.rows() / 2;

  std::cout << "f - (0.5 x H x + b x + f0): " << f - (f0 + _zvalDiffCoeff.dot(z) + 0.5 * z.dot(_zvalDiffHess * z))
            << std::endl;

  double cf = computeCompatibilityEnergy(_baseCWF._omega, z);
  std::cout << "cf - 0.5 x H x: " << cf - 0.5 * z.dot(_zvalCompHess * z) << std::endl;
}

VectorX CWFDecomposition::flatMatrix(const MatrixX& x) {
  auto rows = x.rows();
  auto cols = x.cols();
  VectorX v(rows * cols);

  for (Eigen::Index i = 0; i < rows; i++)
    for (Eigen::Index j = 0; j < cols; j++) {
      v[cols * i + j] = x(i, j);
    }
  return v;
}

MatrixX CWFDecomposition::unFlatVector(const VectorX& v, Eigen::Index cols) {
  auto size = v.size();
  if (size % cols != 0) {
    std::cerr << "conversion failed. vector size: " << size << ", desired cols: " << cols << std::endl;
    exit(EXIT_FAILURE);
  }

  Eigen::Index rows = size / cols;
  MatrixX x(rows, cols);
  for (Eigen::Index i = 0; i < rows; i++)
    for (Eigen::Index j = 0; j < cols; j++) {
      x(i, j) = v[cols * i + j];
    }
  return x;
}

/////////////////////////////////// base mesh update energies //////////////////////////////////////////////////////////
void CWFDecomposition::precomptationForBaseMesh() {
  // normal update. We fixed the _upN
  VectorX upZvals = _upZMat * _baseCWF._zvals;
  _normalWrinkleUpdates = _upN;
  for (int i = 0; i < _upN.rows(); i++) {
    _normalWrinkleUpdates.row(i) *= _upAmp[i] * upZvals[i];
  }

  int nupverts = _upV.rows();
  int nverts = _baseCWF._mesh.GetVertCount();
  SparseMatrixX massMat(3 * nupverts, 3 * nupverts), liftedUpsampleMat(3 * nupverts, 3 * nverts);
  std::vector<TripletX> mT, lT;

  for (int i = 0; i < nupverts; i++) {
    mT.push_back({3 * i, 3 * i, _upVertArea[i]});
    mT.push_back({3 * i + 1, 3 * i + 1, _upVertArea[i]});
    mT.push_back({3 * i + 2, 3 * i + 2, _upVertArea[i]});
  }
  massMat.setFromTriplets(mT.begin(), mT.end());


  for (int k = 0; k < _LoopS0.outerSize(); ++k) {
    for (Eigen::SparseMatrix<Scalar>::InnerIterator it(_LoopS0, k); it; ++it) {
      auto& c = it.value();
      for (int j = 0; j < 3; j++) lT.push_back({TripletX(3 * it.row() + j, 3 * it.col() + j, c)});
    }
  }
  liftedUpsampleMat.setFromTriplets(lT.begin(), lT.end());

  SparseMatrixX tmpMat = liftedUpsampleMat.transpose() * massMat;
  _baseMeshDiffHess = tmpMat * liftedUpsampleMat;

  VectorX diff = flatMatrix(_wrinkledV - _normalWrinkleUpdates);
  _baseMeshDiffCoeff = -tmpMat * diff;
}

double CWFDecomposition::computeDifferenceFromBasemesh(const MatrixX& pos, VectorX* grad, SparseMatrixX* hess) {
  double energy = 0;
  MatrixX upPos = _LoopS0 * pos;

  int nupverts = upPos.rows();

  for (int i = 0; i < nupverts; i++) {
    energy += 0.5 * _upVertArea[i] * (_wrinkledV.row(i) - upPos.row(i) - _normalWrinkleUpdates.row(i)).squaredNorm();
  }
  return energy;
}

void CWFDecomposition::testDifferenceFromBasemesh(const MatrixX& pos) {
  MatrixX pos0 = pos;
  precomputationForPhase();
  precomptationForBaseMesh();

  pos0.setZero();
  double energy0 = computeDifferenceFromBasemesh(pos0, nullptr, nullptr);

  pos0.setRandom();
  double energy1 = computeDifferenceFromBasemesh(pos0, nullptr, nullptr);

  VectorX flatPos = flatMatrix(pos0);
  double energy2 = energy0 + _baseMeshDiffCoeff.dot(flatPos) + 0.5 * flatPos.dot(_baseMeshDiffHess * flatPos);

  std::cout << "f - (0.5 x H x + b x + f0): " << energy1 - energy2 << std::endl;
}
} // namespace ComplexWrinkleField
