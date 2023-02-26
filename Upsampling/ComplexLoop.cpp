#include "../CommonTools.h"
#include "ComplexLoop.h"
#include <iostream>
#include <cassert>
#include <memory>

std::vector<std::complex<double>> ComplexLoop::computeComplexWeight(const std::vector<double>& dthetaList, const std::vector<double>& coordList)
{
	int nPoints = dthetaList.size();
	std::vector<std::complex<double>> complexWeights(nPoints, 0);

	for(int i = 0; i < nPoints; i++)
	{
		complexWeights[i] = coordList[i] * std::complex<double>(std::cos(dthetaList[i]), std::sin(dthetaList[i]));
	}
	return complexWeights;
}

std::vector<std::complex<double>> ComplexLoop::computeComplexWeight(const std::vector<Eigen::Vector3d>& pList, const std::vector<Eigen::Vector3d>& gradThetaList, const std::vector<double>& pWeights)
{
	int nPoints = pList.size();
	std::vector<double> dthetaList(nPoints, 0);

	Eigen::Vector3d p = Eigen::Vector3d::Zero();
	for(int i = 0; i < nPoints; i++)
	{
		p += pWeights[i] * pList[i];
	}

	for(int i = 0; i < nPoints; i++)
	{
		dthetaList[i] = gradThetaList[i].dot(p - pList[i]);
	}
	return computeComplexWeight(dthetaList, pWeights);
}

std::vector<std::complex<double>> ComplexLoop::computeEdgeComplexWeight(const Eigen::VectorXd& omega, const Eigen::Vector2d& bary, int eid)
{
	double edgeOmega = omega[eid];  // one form, refered as theta[e1] - theta[e0]
	double dtheta0 = bary[1] * edgeOmega;
	double dtheta1 = -bary[0] * edgeOmega;

	std::vector<double> dthetaList = {dtheta0, dtheta1};
	std::vector<double> coordList = {bary[0], bary[1]};

	return computeComplexWeight(dthetaList, coordList);
}

std::vector<std::complex<double>> ComplexLoop::computeTriangleComplexWeight(const Eigen::VectorXd& omega, const Eigen::Vector3d& bary, int fid)
{
	std::vector<double> coordList = {bary[0], bary[1], bary[2]};
	std::vector<double> dthetaList(3, 0);
	const std::vector<int>& vertList = _mesh.GetFaceVerts(fid);
	const std::vector<int>& edgeList = _mesh.GetFaceEdges(fid);

	for(int i = 0; i < 3; i++)
	{
		int vid = vertList[i];
		int eid0 = edgeList[i];
		int eid1 = edgeList[(i + 2) % 3];

		double w0 = omega(eid0);
		double w1 = omega(eid1);

		if (vid == _mesh.GetEdgeVerts(eid0)[1])
			w0 *= -1;
		if (vid == _mesh.GetEdgeVerts(eid1)[1])
			w1 *= -1;

		dthetaList[i] = w0 * bary((i + 1) % 3) + w1 * bary((i + 2) % 3);
	}
	return computeComplexWeight(dthetaList, coordList);
}

void ComplexLoop::BuildComplexS0(const Eigen::VectorXd& omega, Eigen::SparseMatrix<std::complex<double>>& A)
{
	int V = _mesh.GetVertCount();
	int E = _mesh.GetEdgeCount();

	std::vector<Eigen::Triplet<std::complex<double>>> T;

	// Even (old) vertices
	for(int vi = 0; vi < V; ++vi)
	{
		if(_mesh.IsVertBoundary(vi))  // boundary vertices
		{
			if(_isFixBnd)
			{
				T.push_back({2 * vi, 2 * vi, 1.0});
				T.push_back({2 * vi + 1, 2 * vi + 1, 1.0});
			}
			else
			{
				std::vector<int> boundary(2);
				boundary[0] = _mesh.GetVertEdges(vi).front();
				boundary[1] = _mesh.GetVertEdges(vi).back();

				std::vector<Eigen::Vector3d> gradthetap(2);
				std::vector<double> coords = { 1. / 2, 1. / 2 };
				std::vector<Eigen::Vector3d> pList(2);
				std::vector<std::vector<int>> edgeVertMap(2, {-1, -1});

				std::vector<std::vector<std::complex<double>>> innerWeights(2);
				std::vector<std::complex<double>> outerWeights(2);

				for (int j = 0; j < boundary.size(); ++j)
				{
					int edge = boundary[j];
					int face = _mesh.GetEdgeFaces(edge)[0];
					int viInface = _mesh.GetVertIndexInFace(face, vi);

					int viInEdge = _mesh.GetVertIndexInEdge(edge, vi);
					int vj = _mesh.GetEdgeVerts(edge)[(viInEdge + 1) % 2];

					int vjInface = _mesh.GetVertIndexInFace(face, vj);

					Eigen::Vector3d bary = Eigen::Vector3d::Zero();
					bary(viInface) = 3. / 4;
					bary(vjInface) = 1. / 4;

					Eigen::Vector2d edgeBary;
					bary[viInEdge] = 3. / 4.;
					bary[1 - viInEdge] = 1. / 4.;

					edgeVertMap[j][viInEdge] = vi;
					edgeVertMap[j][1 - viInEdge] = vj;

					pList[j] = 3. / 4 * _mesh.GetVertPos(vi) + 1. / 4 * _mesh.GetVertPos(vj);
					// grad from vi
					gradthetap[j] = computeBaryGradThetaFromOmegaPerface(omega, face, bary);
					innerWeights[j] = computeEdgeComplexWeight(omega, edgeBary, edge);
				}
				outerWeights = computeComplexWeight(pList, gradthetap, coords);

				for(int j = 0; j < 2; j++)
				{
					for(int k = 0; k < 2; k++)
					{
						std::complex<double> cjk = outerWeights[j] * innerWeights[j][k];
						T.push_back({vi, edgeVertMap[j][k], cjk});
					}
				}
			}
		}
		else        // inner vertices
		{
			const std::vector<int>& vFaces = _mesh.GetVertFaces(vi);
			int nNeiFaces = vFaces.size();

			// Fig5 left [Wang et al. 2006]
			Scalar alpha = 0.375;
			if (nNeiFaces == 3)
				alpha /= 2;
			else
				alpha /= nNeiFaces;

			double beta = nNeiFaces / 2. * alpha;

			std::vector<std::complex<double>> zp(nNeiFaces);
			std::vector<Eigen::Vector3d> gradthetap(nNeiFaces);
			std::vector<double> coords;
			coords.resize(nNeiFaces, 1. / nNeiFaces);
			std::vector<Eigen::Vector3d> pList(nNeiFaces);
			std::vector<std::vector<std::complex<double>>> innerWeights(nNeiFaces);
			std::vector<std::complex<double>> outerWeights(nNeiFaces);
			std::vector<std::vector<int>> faceVertMap(nNeiFaces, {-1, -1, -1});

			for (int k = 0; k < nNeiFaces; ++k)
			{
				int face = vFaces[k];
				int viInface = _mesh.GetVertIndexInFace(face, vi);
				Eigen::Vector3d bary;
				bary.setConstant(beta);
				bary(viInface) = 1 - 2 * beta;

				pList[k] = Eigen::Vector3d::Zero();
				for (int i = 0; i < 3; i++)
				{
					pList[k] += bary(i) * _mesh.GetVertPos(_mesh.GetFaceVerts(face)[i]);
				}
				faceVertMap[k] = _mesh.GetFaceVerts(face);

				innerWeights[k] = computeTriangleComplexWeight(omega, bary, face);
				gradthetap[k] = computeBaryGradThetaFromOmegaPerface(omega, face, bary);
				
			}
			outerWeights = computeComplexWeight(pList, gradthetap, coords);

			for(int j = 0; j < nNeiFaces; j++)
			{
				for(int k = 0; k < 3; k++)
				{
					std::complex<double> cjk = outerWeights[j] * innerWeights[j][k];
					T.push_back({vi, faceVertMap[j][k], cjk});
				}
			}
		}
	}

	// Odd (new) vertices
	for (int edge = 0; edge < E; ++edge)
	{
		int row = edge + V;
		if (_mesh.IsEdgeBoundary(edge))
		{
			Eigen::Vector2d bary;
			bary << 0.5, 0.5;
			std::vector<std::complex<double>> complexWeight = computeEdgeComplexWeight(omega, bary, edge);
			T.push_back({row, _mesh.GetEdgeVerts(edge)[0], complexWeight[0]});
			T.push_back({row, _mesh.GetEdgeVerts(edge)[1], complexWeight[1]});
		}
		else
		{
			std::vector<Eigen::Vector3d> gradthetap(2);
			std::vector<double> coords = { 1. / 2, 1. / 2 };
			std::vector<Eigen::Vector3d> pList(2);
			std::vector<std::vector<std::complex<double>>> innerWeights(2);
			std::vector<std::complex<double>> outerWeights(2);
			std::vector<std::vector<int>> faceVertMap(2, {-1, -1, -1});

			for (int j = 0; j < 2; ++j)
			{
				int face = _mesh.GetEdgeFaces(edge)[j];
				int offset = _mesh.GetEdgeIndexInFace(face, edge);

				Eigen::Vector3d bary;
				bary.setConstant(3. / 8.);
				bary((offset + 2) % 3) = 0.25;

				pList[j] = Eigen::Vector3d::Zero();
				for (int i = 0; i < 3; i++)
				{
					pList[j] += bary(i) * _mesh.GetVertPos(_mesh.GetFaceVerts(face)[i]);
				}
				faceVertMap[j] = _mesh.GetFaceVerts(face);
				innerWeights[j] = computeTriangleComplexWeight(omega, bary, face);
				gradthetap[j] = computeBaryGradThetaFromOmegaPerface(omega, face, bary);
			}
			outerWeights = computeComplexWeight(pList, gradthetap, coords);

			for(int j = 0; j < 2; j++)
			{
				for(int k = 0; k < 3; k++)
				{
					std::complex<double> cjk = outerWeights[j] * innerWeights[j][k];
					T.push_back({row, faceVertMap[j][k], cjk});
				}
			}
		}
	}

	A.resize(V + E, V);
	A.setFromTriplets(T.begin(), T.end());
}

Eigen::Vector3d ComplexLoop::computeGradThetaFromOmegaPerface(const Eigen::VectorXd& omega, int fid, int vInF)
{
	int vid = _mesh.GetFaceVerts(fid)[vInF];
	int eid0 = _mesh.GetFaceEdges(fid)[vInF];
	int eid1 = _mesh.GetFaceEdges(fid)[(vInF + 2) % 3];
	Eigen::Vector3d r0 = _mesh.GetVertPos(_mesh.GetEdgeVerts(eid0)[1]) - _mesh.GetVertPos(_mesh.GetEdgeVerts(eid0)[0]);
	Eigen::Vector3d r1 = _mesh.GetVertPos(_mesh.GetEdgeVerts(eid1)[1]) - _mesh.GetVertPos(_mesh.GetEdgeVerts(eid1)[0]);

	Eigen::Matrix2d Iinv, I;
	I << r0.dot(r0), r0.dot(r1), r1.dot(r0), r1.dot(r1);
	Iinv = I.inverse();

	Eigen::Vector2d rhs;
	double w1 = omega(eid0);
	double w2 = omega(eid1);
	rhs << w1, w2;

	Eigen::Vector2d u = Iinv * rhs;
	return u[0] * r0 + u[1] * r1;
}

Eigen::Vector3d ComplexLoop::computeBaryGradThetaFromOmegaPerface(const Eigen::VectorXd& omega, int fid, const Eigen::Vector3d& bary)
{
	Eigen::Vector3d gradTheta = Eigen::Vector3d::Zero();
	for(int i = 0; i < 3; i++)
	{
		gradTheta += bary[i] * computeGradThetaFromOmegaPerface(omega, fid, i);
	}
	return gradTheta;
}


void ComplexLoop::CWFSubdivide(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, int level)
{
	
	int nverts = _mesh.GetVertCount();
	omegaNew = omega;
	upZvals = zvals;


	MatrixX X;
	_mesh.GetPos(X);
	auto _backupMesh = _mesh;

	Eigen::VectorXd amp(nverts);
	Eigen::VectorXcd zvec(nverts);
	
	for (int i = 0; i < nverts; i++)
	{
		amp(i) = std::abs(zvals[i]);
		zvec[i] = zvals[i];
	}

	


	for (int l = 0; l < level; ++l) 
	{
		SparseMatrixX tmpS0, tmpS1;
		Eigen::SparseMatrix<std::complex<double>> tmpCS0;
		BuildS0(tmpS0);
		BuildS1(tmpS1);
		BuildComplexS0(omegaNew, tmpCS0);

		X = tmpS0 * X;
		amp = tmpS0 * amp;
		zvec = tmpCS0 * zvec;
		omegaNew = tmpS1 * omegaNew;

		std::vector<Vector3> points;
		ConvertToVector3(X, points);

		std::vector< std::vector<int> > edgeToVert;
		GetSubdividedEdges(edgeToVert);

		std::vector< std::vector<int> > faceToVert;
		GetSubdividedFaces(faceToVert);

		_mesh.Populate(points, faceToVert, edgeToVert);
	}
	std::swap(_backupMesh, _mesh);
	upZvals.resize(zvec.rows());
	for (int i = 0; i < upZvals.size(); i++)
		upZvals[i] = zvec[i];
}
