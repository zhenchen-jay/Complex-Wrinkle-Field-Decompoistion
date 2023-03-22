#include "BaseMeshExtraction.h"
#include <igl/principal_curvature.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <Eigen/CholmodSupport>

void basemeshExtraction(const Mesh& wrinkledMesh, MatrixX& basemeshPos, Eigen::MatrixXi& basemeshFaces, MatrixX* isoPos, Eigen::MatrixXi* isoEdges)
{
	// prepare
	Eigen::MatrixXi wrinkledFaceNeighbors, wrinkledFaces;
	Eigen::MatrixXd wrinkledPos;
	wrinkledFaces = wrinkledMesh.GetFace();
	wrinkledPos = wrinkledMesh.GetPos();

	wrinkledFaceNeighbors.resize(wrinkledFaces.rows(), 3);
	for(int i = 0; i < wrinkledFaces.rows(); i++)
	{
		for(int j = 0; j < 3; j++)
		{
			int eid = wrinkledMesh.GetFaceEdges(i)[(j + 1) % 3];
			if(wrinkledMesh.GetEdgeFaces(eid).size() == 1)
				wrinkledFaceNeighbors(i, j) = -1;
			else
				wrinkledFaceNeighbors(i, j) = wrinkledMesh.GetEdgeFaces(eid)[0] != i ? wrinkledMesh.GetEdgeFaces(eid)[0] : wrinkledMesh.GetEdgeFaces(eid)[1];
		}
	}
	// step 1: compute mean curvatures
	Eigen::MatrixXd PD1, PD2;
	Eigen::VectorXd PV1, PV2;
	igl::principal_curvature(wrinkledPos, wrinkledFaces, PD1, PD2, PV1, PV2);
	Eigen::VectorXd H = (PV1 + PV2) / 2;

	// step 2: extract zero iso-pts and iso-lines
	Eigen::MatrixXd isolinePos, splittedWrinkledPos;
	Eigen::MatrixXi isolineEdges, splittedWrinkledFaces;
	extractIsoline(wrinkledPos, wrinkledFaces, wrinkledFaceNeighbors, H, 0, isolinePos, isolineEdges, splittedWrinkledPos, splittedWrinkledFaces);

	if (isoPos)
		*isoPos = isolinePos;
	if (isoEdges)
		*isoEdges = isolineEdges;

	// step 3: build bilaplacian
	int nverts = wrinkledPos.rows();
	int nsplittedverts = splittedWrinkledPos.rows();
	Eigen::VectorXd doubleArea, vertAreaInv;
	igl::doublearea(splittedWrinkledPos, splittedWrinkledFaces, doubleArea);
	vertAreaInv.setZero(nsplittedverts);
	for(int i = 0; i < doubleArea.rows(); i++)
	{
		for(int j = 0; j < 3; j++)
			vertAreaInv[splittedWrinkledFaces(i, j)] += doubleArea[i] / 6.0;
	}
	SparseMatrixX massInv(vertAreaInv.rows(), vertAreaInv.rows());
	std::vector<TripletX> T;
	for(int i = 0; i < vertAreaInv.rows(); i++)
		T.push_back({i, i, vertAreaInv[i]});
	massInv.setFromTriplets(T.begin(), T.end());

	// step 3: update the wrinkle pos
	basemeshPos = splittedWrinkledPos;
	basemeshFaces = splittedWrinkledFaces;
	SparseMatrixX L;
	igl::cotmatrix(splittedWrinkledPos, splittedWrinkledFaces, L);

	// step 4: build bilaplacian (quadratic bending)
	SparseMatrixX BiLap = L * massInv * L, fullBiLap(3 * nsplittedverts, 3 * nsplittedverts);
	//extend BiLap to 3n x 3n so it applies on all three directions
	std::vector<TripletX> Llist;
	for (int k = 0; k < BiLap.outerSize(); k++){
		for (Eigen::SparseMatrix<double>::InnerIterator it(BiLap,k); it; ++it){
			for (int i = 0; i < 3; i++)
				Llist.push_back(Eigen::Triplet<double>(3 * it.row() + i, 3 * it.col() + i, it.value()));
		}
	}
	fullBiLap.setFromTriplets(Llist.begin(), Llist.end());

	// step 5: form the projected matrix
	T.clear();
	for(int i = 0; i < 3 * nverts; i++)
	{
		T.push_back({ i, i, 1 });
	}

	SparseMatrixX projMat(3 * nverts, 3 * nsplittedverts);
	projMat.setFromTriplets(T.begin(), T.end());

	// step 6: solve the linear system to get the solution
	auto mat2vec = [&](const Eigen::MatrixXd& mat)
	{
		Eigen::VectorXd x(mat.rows() * mat.cols());
		for (int i = 0; i < mat.rows(); i++)
		{
			for (int j = 0; j < mat.cols(); j++)
			{
				x[mat.cols() * i + j] = mat(i, j);
			}
		}
		return x;
	};
	auto vec2mat = [&](const Eigen::VectorXd& x, int ncols = 3)
	{
		int nrows = x.size() / 3;
		Eigen::MatrixXd mat(nrows, ncols);

		for (int i = 0; i < mat.rows(); i++)
		{
			for (int j = 0; j < mat.cols(); j++)
			{
				mat(i, j) = x[mat.cols() * i + j];
			}
		}
		return mat;
	};

	Eigen::VectorXd x0 = mat2vec(basemeshPos);
	x0.segment(0, 3 * nverts).setZero();

	SparseMatrixX projHess = projMat * fullBiLap * projMat.transpose(), PDHess = projHess;
	VectorX b = projMat * fullBiLap.transpose() * x0;

	b *= -1;	// tricky choldmod bug

	Eigen::SparseMatrix<double> I(3 * nverts, 3 * nverts);
	I.setIdentity();

	Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver(PDHess);

	double reg = 1e-8;
	while (solver.info() != Eigen::Success)
	{
		PDHess = projHess + reg * I;
		solver.compute(PDHess);
		reg = std::max(2 * reg, 1e-16);

		if (reg > 1e4)
		{
			std::cout << "reg is too large. ||H|| = " << projHess.norm() << std::endl;
			return;
		}
	}
	
	Eigen::VectorXd y = solver.solve(b);
	x0 = x0 + projMat.transpose() * y;
	basemeshPos = vec2mat(x0);

}