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

std::complex<double> ComplexLoop::interpZ(const std::vector<std::complex<double>>& zList, const std::vector<Eigen::Vector3d>& gradThetaList, std::vector<double>& coords, const std::vector<Eigen::Vector3d>& pList)
{
	Eigen::Vector3d P = Eigen::Vector3d::Zero();
	
	int n = pList.size();
	for (int i = 0; i < n; i++)
	{
		P += coords[i] * pList[i];
	}
	Eigen::Vector3d gradThetaAve = Eigen::Vector3d::Zero();
	for (int i = 0; i < n; i++)
	{
		gradThetaAve += coords[i] * gradThetaList[i];
	}

	std::complex<double> z = 0;

	std::complex<double> I = std::complex<double>(0, 1);

	std::vector<double> deltaThetaList(n);
	for (int i = 0; i < n; i++)
	{
		//deltaThetaList[i] = (P - pList[i]).dot(gradThetaList[i]);
		deltaThetaList[i] = (P - pList[i]).dot(gradThetaAve);
	}

	Eigen::VectorXcd dzdalpha;
	dzdalpha.setZero(n);

	for (int i = 0; i < n; i++)
	{
		z += coords[i] * zList[i] * std::complex<double>(std::cos(deltaThetaList[i]), std::sin(deltaThetaList[i]));
	}

	return z;
}

std::complex<double> ComplexLoop::computeZandGradZ(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, int fid, const Eigen::Vector3d& bary, Eigen::Vector3cd* gradz)
{
	Eigen::Matrix3d gradBary;
	Eigen::Vector3d P0 = _mesh.GetVertPos(_mesh.GetFaceVerts(fid)[0]);
	Eigen::Vector3d P1 = _mesh.GetVertPos(_mesh.GetFaceVerts(fid)[1]);
	Eigen::Vector3d P2 = _mesh.GetVertPos(_mesh.GetFaceVerts(fid)[2]);
	
	computeBaryGradient(P0, P1, P2, bary, gradBary);

	std::complex<double> z = 0;

	if (gradz)
	{
		gradz->setZero();
	}
	
	
	std::complex<double> tau = std::complex<double>(0, 1);

	for (int i = 0; i < 3; i++)
	{
		double alphai = bary(i);
		int vid = _mesh.GetFaceVerts(fid)[i];
		int eid0 = _mesh.GetFaceEdges(fid)[i];
		int eid1 = _mesh.GetFaceEdges(fid)[(i + 2) % 3];

		double w1 = omega(eid0);
		double w2 = omega(eid1);

		if (vid == _mesh.GetEdgeVerts(eid0)[1])
			w1 *= -1;
		if (vid == _mesh.GetEdgeVerts(eid1)[1])
			w2 *= -1;

		double deltaTheta = w1 * bary((i + 1) % 3) + w2 * bary((i + 2) % 3);
		std::complex<double> expip = std::complex<double>(std::cos(deltaTheta), std::sin(deltaTheta));

		z += alphai * zvals[vid] * expip;

		if(gradz)
			(*gradz) += zvals[vid] * expip * (gradBary.row(i) + tau * alphai * w1 * gradBary.row((i + 1) % 3) + tau * alphai * w2 * gradBary.row((i + 2) % 3));
	}
	return z;
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

void ComplexLoop::updateLoopedZvals(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, std::vector<std::complex<double>>& upZvals)
{
	int V = _mesh.GetVertCount();
	int E = _mesh.GetEdgeCount();
	
	upZvals.resize(V + E);
	// Even verts
	for (int vi = 0; vi < V; ++vi)
	{
		if (_mesh.IsVertBoundary(vi))
		{
			if(_isFixBnd)
				upZvals[vi] = zvals[vi];
			else
			{
				std::vector<int> boundary(2);
				boundary[0] = _mesh.GetVertEdges(vi).front();
				boundary[1] = _mesh.GetVertEdges(vi).back();

				std::vector<std::complex<double>> zp(2);
				std::vector<Eigen::Vector3d> gradthetap(2);
				std::vector<double> coords = { 1. / 2, 1. / 2 };
				std::vector<Eigen::Vector3d> pList(2);

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

                    pList[j] = 3. / 4 * _mesh.GetVertPos(vi) + 1. / 4 * _mesh.GetVertPos(vj);
                    // grad from vi
                    gradthetap[j] = computeBaryGradThetaFromOmegaPerface(omega, face, bary);
                    zp[j] = computeZandGradZ(omega, zvals, face, bary, NULL);

				}
				upZvals[vi] = interpZ(zp, gradthetap, coords, pList);
			}
			
			
		}
		else
		{
			const std::vector<int>& vFaces = _mesh.GetVertFaces(vi);
			int nNeiFaces = vFaces.size();

			// Fig5 left [Wang et al. 2006]
			Scalar alpha = 0.375;
			if (nNeiFaces == 3) alpha /= 2;
			else                    alpha /= nNeiFaces;

			double beta = nNeiFaces / 2. * alpha;

			std::vector<std::complex<double>> zp(nNeiFaces);
			std::vector<Eigen::Vector3d> gradthetap(nNeiFaces);
			std::vector<double> coords;
			coords.resize(nNeiFaces, 1. / nNeiFaces);
			std::vector<Eigen::Vector3d> pList(nNeiFaces);

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

                zp[k] = computeZandGradZ(omega, zvals, face, bary, NULL);
                gradthetap[k] = computeBaryGradThetaFromOmegaPerface(omega, face, bary);

			}
			upZvals[vi] = interpZ(zp, gradthetap, coords, pList);
		}
	}

	// Odd verts
	for (int edge = 0; edge < E; ++edge)
	{
		int row = edge + V;
		if (_mesh.IsEdgeBoundary(edge))
		{
			int face = _mesh.GetEdgeFaces(edge)[0];
			int eindexFace = _mesh.GetEdgeIndexInFace(face, edge);
			Eigen::Vector3d bary;
			bary.setConstant(0.5);
			bary((eindexFace + 2) % 3) = 0;
            
			upZvals[row] = computeZandGradZ(omega, zvals, face, bary, NULL);
		}
		else
		{
			
			std::vector<std::complex<double>> zp(2);
			std::vector<Eigen::Vector3d> gradthetap(2);
			std::vector<double> coords = { 1. / 2, 1. / 2 };
			std::vector<Eigen::Vector3d> pList(2);

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

                zp[j] = computeZandGradZ(omega, zvals, face, bary, NULL);
                gradthetap[j] = computeBaryGradThetaFromOmegaPerface(omega, face, bary);

			}

			upZvals[row] = interpZ(zp, gradthetap, coords, pList);
		}
	}
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
	
	for (int i = 0; i < nverts; i++)
	{
		amp(i) = std::abs(zvals[i]);
	}

	for (int l = 0; l < level; ++l) 
	{
		SparseMatrixX tmpS0, tmpS1;
		BuildS0(tmpS0);
		BuildS1(tmpS1);

		X = tmpS0 * X;
		amp = tmpS0 * amp;

		std::vector<std::complex<double>> upZvalsNew;
		//std::vector<Eigen::Vector3cd> upGradZvals;

		updateLoopedZvals(omegaNew, upZvals, upZvalsNew);

		//updateLoopedZvalsNew(meshNew, omegaNew, upZvals, upZvalsNew);
		omegaNew = tmpS1 * omegaNew;

		std::vector<Vector3> points;
		ConvertToVector3(X, points);

		std::vector< std::vector<int> > edgeToVert;
		GetSubdividedEdges(edgeToVert);

		std::vector< std::vector<int> > faceToVert;
		GetSubdividedFaces(faceToVert);

		_mesh.Populate(points, faceToVert, edgeToVert);

		upZvals.swap(upZvalsNew);
	}
    std::swap(_backupMesh, _mesh);
}
