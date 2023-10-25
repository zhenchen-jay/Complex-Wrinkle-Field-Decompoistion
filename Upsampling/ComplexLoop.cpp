#include "../CommonTools.h"
#include "ComplexLoop.h"

namespace ComplexWrinkleField {
    std::vector<std::complex<double>> ComplexLoop::ComputeComplexWeight(const std::vector<double>& dthetaList, const std::vector<double>& coordList) const
    {
        int nPoints = dthetaList.size();
        std::vector<std::complex<double>> complexWeights(nPoints, 0);

        for(int i = 0; i < nPoints; i++)
        {
            complexWeights[i] = coordList[i] * std::complex<double>(std::cos(dthetaList[i]), std::sin(dthetaList[i]));
        }
        return complexWeights;
    }

    std::vector<std::complex<double>> ComplexLoop::ComputeComplexWeight(const std::vector<Vector3>& pList, const std::vector<Vector3>& gradThetaList, const std::vector<double>& pWeights) const
    {
        int nPoints = pList.size();
        std::vector<double> dthetaList(nPoints, 0);

        Vector3 p = Vector3::Zero();
        for(int i = 0; i < nPoints; i++)
        {
            p += pWeights[i] * pList[i];
        }

        for(int i = 0; i < nPoints; i++)
        {
            dthetaList[i] = gradThetaList[i].dot(p - pList[i]);
        }
        return ComputeComplexWeight(dthetaList, pWeights);
    }

    std::vector<std::complex<double>> ComplexLoop::ComputeEdgeComplexWeight(const VectorX& omega, const Vector2& bary, int eid) const
    {
        double edgeOmega = omega[eid];  // one form, refered as theta[e1] - theta[e0]
        double dtheta0 = bary[1] * edgeOmega;
        double dtheta1 = -bary[0] * edgeOmega;

        std::vector<double> dthetaList = {dtheta0, dtheta1};
        std::vector<double> coordList = {bary[0], bary[1]};

        return ComputeComplexWeight(dthetaList, coordList);
    }

    std::vector<std::complex<double>> ComplexLoop::ComputeTriangleComplexWeight(const VectorX& omega, const Vector3& bary, int fid) const
    {
        std::vector<double> coordList = {bary[0], bary[1], bary[2]};
        std::vector<double> dthetaList(3, 0);
        const std::vector<int>& vertList = _mesh->GetFaceVerts(fid);
        const std::vector<int>& edgeList = _mesh->GetFaceEdges(fid);

        for(int i = 0; i < 3; i++)
        {
            int vid = vertList[i];
            int eid0 = edgeList[i];
            int eid1 = edgeList[(i + 2) % 3];

            double w0 = omega(eid0);
            double w1 = omega(eid1);

            if (vid == _mesh->GetEdgeVerts(eid0)[1])
                w0 *= -1;
            if (vid == _mesh->GetEdgeVerts(eid1)[1])
                w1 *= -1;

            dthetaList[i] = w0 * bary((i + 1) % 3) + w1 * bary((i + 2) % 3);
        }
        return ComputeComplexWeight(dthetaList, coordList);
    }

    void ComplexLoop::BuildComplexS0(const VectorX& omega, ComplexSparseMatrixX& A) const
    {
        int V = _mesh->GetVertCount();
        int E = _mesh->GetEdgeCount();

        std::vector<Eigen::Triplet<std::complex<double>>> T;

        // Even (old) vertices
        for(int vi = 0; vi < V; ++vi)
        {
            if(_mesh->IsVertBoundary(vi))  // boundary vertices
            {
                if(_isFixBnd)
                {
                    T.push_back({vi, vi, 1.0});
                }
                else
                {
                    std::vector<int> boundary(2);
                    boundary[0] = _mesh->GetVertEdges(vi).front();
                    boundary[1] = _mesh->GetVertEdges(vi).back();

                    std::vector<Vector3> gradthetap(2);
                    std::vector<double> coords = { 1. / 2, 1. / 2 };
                    std::vector<Vector3> pList(2);
                    std::vector<std::vector<int>> edgeVertMap(2, {-1, -1});

                    std::vector<std::vector<std::complex<double>>> innerWeights(2);
                    std::vector<std::complex<double>> outerWeights(2);

                    for (int j = 0; j < boundary.size(); ++j)
                    {
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
                        gradthetap[j] = ComputeBaryGradThetaFromOmegaPerface(omega, face, bary);
                        innerWeights[j] = ComputeEdgeComplexWeight(omega, edgeBary, edge);
                    }
                    outerWeights = ComputeComplexWeight(pList, gradthetap, coords);

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
                const std::vector<int>& vFaces = _mesh->GetVertFaces(vi);
                int nNeiFaces = vFaces.size();

                // Fig5 left [Wang et al. 2006]
                Scalar alpha = 0.375;
                if (nNeiFaces == 3)
                    alpha /= 2;
                else
                    alpha /= nNeiFaces;

                double beta = nNeiFaces / 2. * alpha;

                std::vector<std::complex<double>> zp(nNeiFaces);
                std::vector<Vector3> gradthetap(nNeiFaces);
                std::vector<double> coords;
                coords.resize(nNeiFaces, 1. / nNeiFaces);
                std::vector<Vector3> pList(nNeiFaces);
                std::vector<std::vector<std::complex<double>>> innerWeights(nNeiFaces);
                std::vector<std::complex<double>> outerWeights(nNeiFaces);
                std::vector<std::vector<int>> faceVertMap(nNeiFaces, {-1, -1, -1});

                for (int k = 0; k < nNeiFaces; ++k)
                {
                    int face = vFaces[k];
                    int viInface = _mesh->GetVertIndexInFace(face, vi);
                    Vector3 bary;
                    bary.setConstant(beta);
                    bary(viInface) = 1 - 2 * beta;

                    pList[k] = Vector3::Zero();
                    for (int i = 0; i < 3; i++)
                    {
                        pList[k] += bary(i) * _mesh->GetVertPos(_mesh->GetFaceVerts(face)[i]);
                    }
                    faceVertMap[k] = _mesh->GetFaceVerts(face);

                    innerWeights[k] = ComputeTriangleComplexWeight(omega, bary, face);
                    gradthetap[k] = ComputeBaryGradThetaFromOmegaPerface(omega, face, bary);

                }
                outerWeights = ComputeComplexWeight(pList, gradthetap, coords);

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
            if (_mesh->IsEdgeBoundary(edge))
            {
                Vector2 bary;
                bary << 0.5, 0.5;
                std::vector<std::complex<double>> complexWeight = ComputeEdgeComplexWeight(omega, bary, edge);
                T.push_back({row, _mesh->GetEdgeVerts(edge)[0], complexWeight[0]});
                T.push_back({row, _mesh->GetEdgeVerts(edge)[1], complexWeight[1]});
            }
            else
            {
                std::vector<Vector3> gradthetap(2);
                std::vector<double> coords = { 1. / 2, 1. / 2 };
                std::vector<Vector3> pList(2);
                std::vector<std::vector<std::complex<double>>> innerWeights(2);
                std::vector<std::complex<double>> outerWeights(2);
                std::vector<std::vector<int>> faceVertMap(2, {-1, -1, -1});

                for (int j = 0; j < 2; ++j)
                {
                    int face = _mesh->GetEdgeFaces(edge)[j];
                    int offset = _mesh->GetEdgeIndexInFace(face, edge);

                    Vector3 bary;
                    bary.setConstant(3. / 8.);
                    bary((offset + 2) % 3) = 0.25;

                    pList[j] = Vector3::Zero();
                    for (int i = 0; i < 3; i++)
                    {
                        pList[j] += bary(i) * _mesh->GetVertPos(_mesh->GetFaceVerts(face)[i]);
                    }
                    faceVertMap[j] = _mesh->GetFaceVerts(face);
                    innerWeights[j] = ComputeTriangleComplexWeight(omega, bary, face);
                    gradthetap[j] = ComputeBaryGradThetaFromOmegaPerface(omega, face, bary);
                }
                outerWeights = ComputeComplexWeight(pList, gradthetap, coords);

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

    Vector3 ComplexLoop::ComputeGradThetaFromOmegaPerface(const VectorX& omega, int fid, int vInF) const
    {
        int vid = _mesh->GetFaceVerts(fid)[vInF];
        int eid0 = _mesh->GetFaceEdges(fid)[vInF];
        int eid1 = _mesh->GetFaceEdges(fid)[(vInF + 2) % 3];
        Vector3 r0 = _mesh->GetVertPos(_mesh->GetEdgeVerts(eid0)[1]) - _mesh->GetVertPos(_mesh->GetEdgeVerts(eid0)[0]);
        Vector3 r1 = _mesh->GetVertPos(_mesh->GetEdgeVerts(eid1)[1]) - _mesh->GetVertPos(_mesh->GetEdgeVerts(eid1)[0]);

        Eigen::Matrix2d Iinv, I;
        I << r0.dot(r0), r0.dot(r1), r1.dot(r0), r1.dot(r1);
        Iinv = I.inverse();

        Vector2 rhs;
        double w1 = omega(eid0);
        double w2 = omega(eid1);
        rhs << w1, w2;

        Vector2 u = Iinv * rhs;
        return u[0] * r0 + u[1] * r1;
    }

    Vector3 ComplexLoop::ComputeBaryGradThetaFromOmegaPerface(const VectorX& omega, int fid, const Vector3& bary) const
    {
        Vector3 gradTheta = Vector3::Zero();
        for(int i = 0; i < 3; i++)
        {
            gradTheta += bary[i] * ComputeGradThetaFromOmegaPerface(omega, fid, i);
        }
        return gradTheta;
    }
}

