#include "BaseMeshExtraction.h"
#include <Eigen/CholmodSupport>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/harmonic.h>
#include <igl/principal_curvature.h>

namespace ComplexWrinkleField {
int getBndProjectionMatrix(const MatrixX& pos, const Eigen::MatrixXi& faces, SparseMatrixX& bndProjM,
                           SparseMatrixX* interiorProjM, std::vector<std::vector<int>>* bndVids,
                           std::vector<bool>* bndFlags) {
  int nverts = pos.rows();
  std::vector<TripletX> bndT, interiorT;
  std::vector<std::vector<int>> L;
  igl::boundary_loop(faces, L);
  std::vector<bool> flags(nverts, false);

  for (auto& bnd : L) {
    for (auto& vid : bnd) flags[vid] = true;
  }


  int nbnds = 0, ninterior = 0;
  for (int i = 0; i < nverts; i++) {
    if (flags[i]) {
      for (int j = 0; j < 3; j++) {
        bndT.push_back({3 * nbnds + j, 3 * i + j, 1.0});
      }
      nbnds++;
    } else {
      if (interiorProjM) {
        for (int j = 0; j < 3; j++) {
          interiorT.push_back({3 * ninterior + j, 3 * i + j, 1.0});
        }
        ninterior++;
      }
    }
  }
  bndProjM.resize(3 * nbnds, 3 * nverts);
  bndProjM.setFromTriplets(bndT.begin(), bndT.end());

  if (interiorProjM) {
    interiorProjM->resize(3 * ninterior, 3 * nverts);
    interiorProjM->setFromTriplets(interiorT.begin(), interiorT.end());
  }

  if (bndVids) *bndVids = L;
  if (bndFlags) *bndFlags = flags;
  return nbnds;
}

void basemeshExtraction(const Mesh& wrinkledMesh, MatrixX& basemeshPos, Eigen::MatrixXi& basemeshFaces, MatrixX* isoPos,
                        Eigen::MatrixXi* isoEdges) {
  // prepare
  Eigen::MatrixXi wrinkledFaceNeighbors, wrinkledFaces;
  Eigen::MatrixXd wrinkledPos;
  wrinkledFaces = wrinkledMesh.GetFace();
  wrinkledPos = wrinkledMesh.GetPos();

  wrinkledFaceNeighbors.resize(wrinkledFaces.rows(), 3);
  for (int i = 0; i < wrinkledFaces.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      int eid = wrinkledMesh.GetFaceEdges(i)[(j + 1) % 3];
      if (wrinkledMesh.GetEdgeFaces(eid).size() == 1)
        wrinkledFaceNeighbors(i, j) = -1;
      else
        wrinkledFaceNeighbors(i, j) = wrinkledMesh.GetEdgeFaces(eid)[0] != i ? wrinkledMesh.GetEdgeFaces(eid)[0]
                                                                             : wrinkledMesh.GetEdgeFaces(eid)[1];
    }
  }

  auto mat2vec = [&](const Eigen::MatrixXd& mat) {
    Eigen::VectorXd x(mat.rows() * mat.cols());
    for (int i = 0; i < mat.rows(); i++) {
      for (int j = 0; j < mat.cols(); j++) {
        x[mat.cols() * i + j] = mat(i, j);
      }
    }
    return x;
  };
  auto vec2mat = [&](const Eigen::VectorXd& x, int ncols = 3) {
    int nrows = x.size() / 3;
    Eigen::MatrixXd mat(nrows, ncols);

    for (int i = 0; i < mat.rows(); i++) {
      for (int j = 0; j < mat.cols(); j++) {
        mat(i, j) = x[mat.cols() * i + j];
      }
    }
    return mat;
  };

  // step 1: compute mean curvatures
  Eigen::MatrixXd PD1, PD2;
  Eigen::VectorXd PV1, PV2, H;
  igl::principal_curvature(wrinkledPos, wrinkledFaces, PD1, PD2, PV1, PV2);
  H = (PV1 + PV2) / 2;

  // step 2: extract zero iso-pts and iso-lines
  Eigen::MatrixXd isolinePos, splittedWrinkledPos;
  Eigen::MatrixXi isolineEdges, splittedWrinkledFaces;
  extractIsoline(wrinkledPos, wrinkledFaces, wrinkledFaceNeighbors, H, 0, isolinePos, isolineEdges, splittedWrinkledPos,
                 splittedWrinkledFaces);

  if (isoPos) *isoPos = isolinePos;
  if (isoEdges) *isoEdges = isolineEdges;

  // step 3: build bilaplacian
  int nverts = wrinkledPos.rows();
  int nsplittedverts = splittedWrinkledPos.rows();
  Eigen::VectorXd doubleArea, vertAreaInv;
  igl::doublearea(splittedWrinkledPos, splittedWrinkledFaces, doubleArea);
  vertAreaInv.setZero(nsplittedverts);
  for (int i = 0; i < doubleArea.rows(); i++) {
    for (int j = 0; j < 3; j++) vertAreaInv[splittedWrinkledFaces(i, j)] += doubleArea[i] / 6.0;
  }
  SparseMatrixX massInv(vertAreaInv.rows(), vertAreaInv.rows());
  std::vector<TripletX> T;
  for (int i = 0; i < vertAreaInv.rows(); i++) T.push_back({i, i, vertAreaInv[i]});
  massInv.setFromTriplets(T.begin(), T.end());

  // step 3: update the wrinkle pos
  basemeshPos = splittedWrinkledPos;
  basemeshFaces = splittedWrinkledFaces;
  SparseMatrixX L;
  igl::cotmatrix(splittedWrinkledPos, splittedWrinkledFaces, L);

  // step 4: build bilaplacian (quadratic bending)
  SparseMatrixX BiLap = L * massInv * L, fullBiLap(3 * nsplittedverts, 3 * nsplittedverts);
  // extend BiLap to 3n x 3n so it applies on all three directions
  std::vector<TripletX> Llist;
  for (int k = 0; k < BiLap.outerSize(); k++) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(BiLap, k); it; ++it) {
      for (int i = 0; i < 3; i++)
        Llist.push_back(Eigen::Triplet<double>(3 * it.row() + i, 3 * it.col() + i, it.value()));
    }
  }
  fullBiLap.setFromTriplets(Llist.begin(), Llist.end());

  // step 5: form the projection matrices
  // step 5.1: wrinkled position to extended wrinkles
  T.clear();
  for (int i = 0; i < 3 * nverts; i++) {
    T.push_back({i, i, 1});
  }
  SparseMatrixX projMat(3 * nverts, 3 * nsplittedverts);
  projMat.setFromTriplets(T.begin(), T.end());

  // step 5.2: boundary and interior projection
  SparseMatrixX bndProjMat, interiorProjMat;
  //	std::vector<int> bndVids;
  std::vector<bool> bndFlags;
  int nbnds = getBndProjectionMatrix(wrinkledPos, wrinkledFaces, bndProjMat, &interiorProjMat, nullptr, &bndFlags);
  // test projection matrix
  //	VectorX testPosVec = mat2vec(wrinkledPos);
  //	VectorX bndPosVec = bndProjMat.transpose() * bndProjMat * testPosVec;
  //	VectorX interiorPosVec = interiorProjMat.transpose() * interiorProjMat * testPosVec;
  //	std::cout << testPosVec.norm() << " " << (testPosVec - bndPosVec - interiorPosVec).norm() << std::endl;
  //	VectorX bndPosVec1(3 * bndVids.size());
  //    int tmpRow = 0;
  //	for (int i = 0; i < bndFlags.size(); i++)
  //	{
  //        if(bndFlags[i])
  //        {
  //            bndPosVec1.segment<3>(3 * tmpRow) << wrinkledPos(i, 0), wrinkledPos(i, 1), wrinkledPos(i, 2);
  //            tmpRow++;
  //        }
  //
  //	}
  //    tmpRow = 0;
  //	VectorX bndPosVec2(3 * bndVids.size());
  //    for (int i = 0; i < bndFlags.size(); i++)
  //    {
  //        if(bndFlags[i])
  //        {
  //            bndPosVec2.segment<3>(3 * tmpRow) = testPosVec.segment<3>(3 * i);
  //            tmpRow++;
  //        }
  //	}
  //	VectorX projBndVec = bndProjMat * testPosVec;
  //	std::cout << "conversion error: " << (bndPosVec1 - projBndVec).norm() << ", " << (bndPosVec1 -
  //bndPosVec2).norm() << ", " << (bndPosVec2 - projBndVec).norm() << std::endl;


  // step 5.3: form the boundary normal matrix
  MatrixX wrinkleNormals;
  igl::per_vertex_normals(wrinkledPos, wrinkledFaces, wrinkleNormals);
  SparseMatrixX bndNormalMat(3 * nbnds, nbnds);
  std::vector<TripletX> normalT;
  int bndRowId = 0;
  for (int i = 0; i < nverts; i++) {
    if (bndFlags[i]) {
      for (int j = 0; j < 3; j++) {
        normalT.push_back({3 * bndRowId + j, bndRowId, wrinkleNormals(i, j)});
      }
      bndRowId++;
    }
  }
  bndNormalMat.setFromTriplets(normalT.begin(), normalT.end());
  // test bnd normal matrix
  //	VectorX perterb(nbnds);
  //	perterb.setRandom();
  //	MatrixX perterbWrinklePos = wrinkledPos;
  //    bndRowId = 0;
  //	for (int i = 0; i < nverts; i++)
  //	{
  //        if(bndFlags[i])
  //        {
  //            perterbWrinklePos.row(i) = wrinkledPos.row(i) + perterb[bndRowId] * wrinkleNormals.row(i);
  //            bndRowId++;
  //        }
  //
  //	}
  //	VectorX perterbY2 = bndProjMat * mat2vec(perterbWrinklePos);
  //	VectorX Y20 = bndProjMat * mat2vec(wrinkledPos);
  //	VectorX Y2 = Y20 + bndNormalMat * perterb;
  //	VectorX Y21 = Y20;
  //
  //	std::cout << "error: " << (perterbY2 - Y2).norm() << std::endl;
  //    bndRowId = 0;
  //    for (int i = 0; i < nverts; i++)
  //    {
  //        if(bndFlags[i])
  //        {
  //            Y21.segment<3>(3 * bndRowId) = Y20.segment<3>(3 * bndRowId) + perterb[bndRowId] *
  //            wrinkleNormals.row(i).segment<3>(0).transpose(); bndRowId++;
  //        }
  //	}
  //	std::cout << "error: " << (perterbY2 - Y21).norm() << std::endl;

  // step 5.4: final projection matrix
  SparseMatrixX finalProjMatInterior = interiorProjMat * projMat;                  // interior part
  SparseMatrixX finalProjMatBnd = bndNormalMat.transpose() * bndProjMat * projMat; // boundary part
  SparseMatrixX finalProjMat(finalProjMatInterior.rows() + finalProjMatBnd.rows(), finalProjMatInterior.cols());
  T.clear();

  for (int k = 0; k < finalProjMatInterior.outerSize(); k++) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(finalProjMatInterior, k); it; ++it) {
      T.push_back(TripletX(it.row(), it.col(), it.value()));
    }
  }
  for (int k = 0; k < finalProjMatBnd.outerSize(); k++) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(finalProjMatBnd, k); it; ++it) {
      T.push_back(TripletX(it.row() + finalProjMatInterior.rows(), it.col(), it.value()));
    }
  }
  finalProjMat.setFromTriplets(T.begin(), T.end());

  // step 6: build the projected hessian and coeff
  SparseMatrixX hess = finalProjMat * fullBiLap * finalProjMat.transpose(), PDHess = hess;
  VectorX flatPos = mat2vec(wrinkledPos);

  Eigen::VectorXd x0 = mat2vec(basemeshPos);
  x0.segment(0, 3 * nverts).setZero();
  x0 = x0 + projMat.transpose() * bndProjMat.transpose() * bndProjMat * flatPos;
  VectorX b = finalProjMat * fullBiLap.transpose() * x0;

  b *= -1; // tricky choldmod bug

  Eigen::SparseMatrix<double> I = hess;
  I.setIdentity();

  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver(PDHess);

  double reg = 1e-8;
  while (solver.info() != Eigen::Success) {
    PDHess = hess + reg * I;
    solver.compute(PDHess);
    reg = std::max(2 * reg, 1e-16);

    if (reg > 1e4) {
      std::cout << "reg is too large. ||H|| = " << hess.norm() << std::endl;
      return;
    }
  }
  Eigen::VectorXd y = solver.solve(b);
  x0 = x0 + finalProjMat.transpose() * y;
  basemeshPos = vec2mat(x0);

  // step 6: solve the linear system to get the solution

  //
  //	Eigen::VectorXd x0 = mat2vec(basemeshPos);
  //	x0.segment(0, 3 * nverts).setZero();
  //
  //	SparseMatrixX projHess = projMat * fullBiLap * projMat.transpose(), PDHess = projHess;
  //	VectorX b = projMat * fullBiLap.transpose() * x0;
  //
  //	b *= -1;	// tricky choldmod bug
  //
  //	Eigen::SparseMatrix<double> I(3 * nverts, 3 * nverts);
  //	I.setIdentity();
  //
  //	Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver(PDHess);
  //
  //	double reg = 1e-8;
  //	while (solver.info() != Eigen::Success)
  //	{
  //		PDHess = projHess + reg * I;
  //		solver.compute(PDHess);
  //		reg = std::max(2 * reg, 1e-16);
  //
  //		if (reg > 1e4)
  //		{
  //			std::cout << "reg is too large. ||H|| = " << projHess.norm() << std::endl;
  //			return;
  //		}
  //	}
  //
  //	Eigen::VectorXd y = solver.solve(b);
  //	x0 = x0 + projMat.transpose() * y;
  //	basemeshPos = vec2mat(x0);
  //
  //    basemeshPos = vec2mat(y);
  //    basemeshFaces = wrinkledFaces;
}

void basemeshHarmonicExtraction(const Mesh& wrinkledMesh, MatrixX& basemeshPos, Eigen::MatrixXi& basemeshFaces,
                                MatrixX* isoPos, Eigen::MatrixXi* isoEdges) {
  // prepare
  Eigen::MatrixXi wrinkledFaceNeighbors, wrinkledFaces;
  Eigen::MatrixXd wrinkledPos;
  wrinkledFaces = wrinkledMesh.GetFace();
  wrinkledPos = wrinkledMesh.GetPos();

  wrinkledFaceNeighbors.resize(wrinkledFaces.rows(), 3);
  for (int i = 0; i < wrinkledFaces.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      int eid = wrinkledMesh.GetFaceEdges(i)[(j + 1) % 3];
      if (wrinkledMesh.GetEdgeFaces(eid).size() == 1)
        wrinkledFaceNeighbors(i, j) = -1;
      else
        wrinkledFaceNeighbors(i, j) = wrinkledMesh.GetEdgeFaces(eid)[0] != i ? wrinkledMesh.GetEdgeFaces(eid)[0]
                                                                             : wrinkledMesh.GetEdgeFaces(eid)[1];
    }
  }

  // step 1: compute mean curvatures
  Eigen::MatrixXd PD1, PD2;
  Eigen::VectorXd PV1, PV2, H;
  igl::principal_curvature(wrinkledPos, wrinkledFaces, PD1, PD2, PV1, PV2);
  H = (PV1 + PV2) / 2;

  // step 2: extract zero iso-pts and iso-lines
  Eigen::MatrixXd isolinePos, splittedWrinkledPos;
  Eigen::MatrixXi isolineEdges, splittedWrinkledFaces;
  extractIsoline(wrinkledPos, wrinkledFaces, wrinkledFaceNeighbors, H, 0, isolinePos, isolineEdges, splittedWrinkledPos,
                 splittedWrinkledFaces);

  std::vector<std::vector<int>> allBndVids;
  igl::boundary_loop(wrinkledFaces, allBndVids);

  std::vector<int> bndVids; // merge all boundaries
  for (auto bnd : allBndVids) {
    bndVids.insert(bndVids.end(), bnd.begin(), bnd.end());
  }

  // step 3: harmonic
  int nisoPts = splittedWrinkledPos.rows() - wrinkledPos.rows();
  Eigen::VectorXi bc(nisoPts + bndVids.size());
  int nsplittedVerts = splittedWrinkledPos.rows();
  int nverts = wrinkledPos.rows();

  for (int i = nverts; i < nsplittedVerts; i++) {
    bc(i - nverts) = i;
  }

  Eigen::MatrixXd fixedPos = isolinePos;
  fixedPos.conservativeResize(bndVids.size() + nisoPts, Eigen::NoChange);

  for (int i = 0; i < bndVids.size(); i++) {
    bc(nisoPts + i) = bndVids[i];
    fixedPos.row(nisoPts + i) = wrinkledPos.row(bndVids[i]);
  }

  igl::harmonic(splittedWrinkledPos, splittedWrinkledFaces, bc, fixedPos, 2, basemeshPos);
  basemeshFaces = splittedWrinkledFaces;
}
} // namespace ComplexWrinkleField
