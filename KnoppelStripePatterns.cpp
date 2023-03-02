#include "KnoppelStripePatterns.h"
#include <igl/cotmatrix.h>
#include <SymGEigsShiftSolver.h>
#include <MatOp/SparseCholesky.h>
#include <Eigen/CholmodSupport>
#include <MatOp/SparseSymShiftSolve.h>
#include <iostream>


void computeEdgeMatrix(const Mesh& mesh, const VectorX& edgeW, const VectorX& edgeWeight, const int nverts, SparseMatrixX& A)
{
	std::vector<TripletX> AT;
	int nedges = mesh.GetEdgeCount();

	for(int i = 0; i < nedges; i++)
	{
		int vid0 = mesh.GetEdgeVerts(i)[0];
		int vid1 = mesh.GetEdgeVerts(i)[1];

		AT.push_back({2 * vid0, 2 * vid0, edgeWeight(i)});
		AT.push_back({2 * vid0 + 1, 2 * vid0 + 1, edgeWeight(i)});

		AT.push_back({2 * vid1, 2 * vid1, edgeWeight(i)});
		AT.push_back({2 * vid1 + 1, 2 * vid1 + 1, edgeWeight(i)});

		std::complex<double> expw0 = std::complex<double>(std::cos(edgeW(i)), std::sin(edgeW(i)));

		AT.push_back({2 * vid0, 2 * vid1, -edgeWeight(i) * (expw0.real())});
		AT.push_back({2 * vid0 + 1, 2 * vid1, -edgeWeight(i) * (-expw0.imag())});
		AT.push_back({2 * vid0, 2 * vid1 + 1, -edgeWeight(i) * (expw0.imag())});
		AT.push_back({2 * vid0 + 1, 2 * vid1 + 1, -edgeWeight(i) * (expw0.real())});

		AT.push_back({ 2 * vid1, 2 * vid0, -edgeWeight(i) * (expw0.real()) });
		AT.push_back({ 2 * vid1, 2 * vid0 + 1, -edgeWeight(i) * (-expw0.imag()) });
		AT.push_back({ 2 * vid1 + 1, 2 * vid0, -edgeWeight(i) * (expw0.imag()) });
		AT.push_back({ 2 * vid1 + 1, 2 * vid0 + 1, -edgeWeight(i) * (expw0.real()) });

	}
	A.resize(2 * nverts, 2 * nverts);
	A.setFromTriplets(AT.begin(), AT.end());
}

void computeEdgeMatrixGivenMag(const Mesh& mesh, const VectorX& edgeW, const VectorX& vertAmp, const VectorX& edgeWeight, const int nverts, SparseMatrixX& A)
{
	std::vector<TripletX> AT;
	int nedges = mesh.GetEdgeCount();

	for (int i = 0; i < nedges; i++) {
        int vid0 = mesh.GetEdgeVerts(i)[0];
        int vid1 = mesh.GetEdgeVerts(i)[1];

		double r0 = vertAmp(vid0);
		double r1 = vertAmp(vid1);

		std::complex<double> expw0 = std::complex<double>(std::cos(edgeW(i)), std::sin(edgeW(i)));


		AT.push_back({2 * vid0, 2 * vid0, r1 * r1 * edgeWeight(i)});
		AT.push_back({2 * vid0 + 1, 2 * vid0 + 1, r1 * r1 * edgeWeight(i)});

		AT.push_back({2 * vid1, 2 * vid1, r0 * r0 * edgeWeight(i)});
		AT.push_back({2 * vid1 + 1, 2 * vid1 + 1, r0 * r0 * edgeWeight(i)});


		AT.push_back({2 * vid0, 2 * vid1, -edgeWeight(i) * (expw0.real()) * r0 * r1});
		AT.push_back({2 * vid0 + 1, 2 * vid1, -edgeWeight(i) * (-expw0.imag()) * r0 * r1});
		AT.push_back({2 * vid0, 2 * vid1 + 1, -edgeWeight(i) * (expw0.imag()) * r0 * r1});
		AT.push_back({2 * vid0 + 1, 2 * vid1 + 1, -edgeWeight(i) * (expw0.real()) * r0 * r1});

		AT.push_back({2 * vid1, 2 * vid0, -edgeWeight(i) * (expw0.real()) * r0 * r1});
		AT.push_back({2 * vid1, 2 * vid0 + 1, -edgeWeight(i) * (-expw0.imag()) * r0 * r1});
		AT.push_back({2 * vid1 + 1, 2 * vid0, -edgeWeight(i) * (expw0.imag()) * r0 * r1});
		AT.push_back({2 * vid1 + 1, 2 * vid0 + 1, -edgeWeight(i) * (expw0.real()) * r0 * r1});
	}
	A.resize(2 * nverts, 2 * nverts);
	A.setFromTriplets(AT.begin(), AT.end());
}

void roundZvalsFromEdgeOmega(const Mesh& mesh, const VectorX& edgeW, const VectorX& edgeWeight, const VectorX& vertArea, int nverts, ComplexVectorX& zvals)
{
	std::vector<TripletX> BT;
	int nfaces = mesh.GetFaceCount();
	int nedges = mesh.GetEdgeCount();

	for (int i = 0; i < nverts; i++)
	{
		BT.push_back({ 2 * i, 2 * i, vertArea(i) });
		BT.push_back({ 2 * i + 1, 2 * i + 1, vertArea(i) });
	}
	
	
	SparseMatrixX A;
	computeEdgeMatrix(mesh, edgeW, edgeWeight, nverts, A);

	SparseMatrixX B(2 * nverts, 2 * nverts);
	B.setFromTriplets(BT.begin(), BT.end());

	Spectra::SymShiftInvert<double> op(A, B);
	Spectra::SparseSymMatProd<double> Bop(B);
	Spectra::SymGEigsShiftSolver<Spectra::SymShiftInvert<double>, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, 1, 6, -1e-6);
	geigs.init();
	int nconv = geigs.compute(Spectra::SortRule::LargestMagn, 1e6);

	Eigen::VectorXd evalues;
	Eigen::MatrixXd evecs;

	evalues = geigs.eigenvalues();
	evecs = geigs.eigenvectors();
	if (nconv != 1 || geigs.info() != Spectra::CompInfo::Successful)
	{
		std::cout << "Eigensolver failed to converge!!" << std::endl;
	}

	std::cout << "Eigenvalue is " << evalues[0] << std::endl;

	zvals.resize(nverts);
	for(int i = 0; i < nverts; i++)
	{
		zvals[i] = std::complex<double>(evecs(2 * i, 0), evecs(2 * i + 1, 0));
        zvals[i] = std::complex<double>(std::cos(std::arg(zvals[i])), std::sin(std::arg(zvals[i])));
	}
}

void roundZvalsFromEdgeOmegaVertexMag(const Mesh& mesh, const VectorX& edgeW, const VectorX& vertAmp, const VectorX& edgeWeight, const VectorX& vertArea, int nverts, ComplexVectorX& zvals)
{
	std::vector<TripletX> BT;
	int nfaces = mesh.GetFaceCount();
	int nedges = mesh.GetEdgeCount();

	for (int i = 0; i < nverts; i++)
	{
		BT.push_back({ 2 * i, 2 * i, vertArea(i) });
		BT.push_back({ 2 * i + 1, 2 * i + 1, vertArea(i) });
	}

	SparseMatrixX A;
	computeEdgeMatrixGivenMag(mesh, edgeW, vertAmp, edgeWeight, nverts, A);

	Eigen::CholmodSupernodalLLT<SparseMatrixX> solver;
	SparseMatrixX I = A;

    I.setIdentity();
	double eps = 1e-16;
    SparseMatrixX tmpA = A + eps * I;
	solver.compute(tmpA);
	while(solver.info() != Eigen::Success)
	{
		std::cout << "matrix is not PD after adding "<< eps << " * I" << std::endl;
		solver.compute(tmpA);
		eps *= 2;
        tmpA = A + eps * I;
	}

	SparseMatrixX B(2 * nverts, 2 * nverts);
	B.setFromTriplets(BT.begin(), BT.end());
	//B.setIdentity();

	Spectra::SymShiftInvert<Scalar> op(A, B);
	Spectra::SparseSymMatProd<Scalar> Bop(B);
	Spectra::SymGEigsShiftSolver<Spectra::SymShiftInvert<Scalar>, Spectra::SparseSymMatProd<Scalar>, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, 1, 6, -2 * eps);
	
	geigs.init();
	int nconv = geigs.compute(Spectra::SortRule::LargestMagn, 1e6);

	Eigen::VectorXd evalues;
	Eigen::MatrixXd evecs;

	evalues = geigs.eigenvalues();
	evecs = geigs.eigenvectors();
	if (nconv != 1 || geigs.info() != Spectra::CompInfo::Successful)
	{
		std::cout << "Eigensolver failed to converge!!" << std::endl;
		exit(1);
	}

	std::cout << "Eigenvalue is " << evalues[0] << std::endl;

	zvals.resize(nverts);
	for(int i = 0; i < nverts; i++)
	{
		zvals[i] = std::complex<double>(evecs(2 * i, 0), evecs(2 * i + 1, 0));
		zvals[i] = vertAmp(i) * std::complex<double>(std::cos(std::arg(zvals[i])), std::sin(std::arg(zvals[i])));
	}
}



