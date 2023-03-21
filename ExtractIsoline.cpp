#include "ExtractIsoline.h"
#include <igl/remove_duplicate_vertices.h>

void extractIsopoints(const Mesh& mesh, const Eigen::VectorXd& func, double isoVal, MatrixX& isoV)
{
    //Constants
    Eigen::MatrixXd V = mesh.GetPos();
    Eigen::MatrixXi F = mesh.GetFace();
    const int dim = V.cols();
    assert(dim==2 || dim==3);
    const int nVerts = V.rows();
    assert(func.rows() == nVerts &&
           "There must be as many function entries as vertices");

    const int nedges = mesh.GetEdgeCount();
    int npts = 0;
    isoV.resize(nedges, 3);
    for(int e = 0; e < nedges; e++)
    {
        int ev0 = mesh.GetEdgeVerts(e)[0];
        int ev1 = mesh.GetEdgeVerts(e)[1];
        const Scalar z1 = func(ev0), z2 = func(ev1);
        double t = (isoVal - z1) / (z2 - z1);
        if (t >= 0 && t <= 1)
        {
            isoV.row(npts) = (1 - t) * V.row(ev0) + t * V.row(ev1);
            npts++;
        }
    }

    isoV.conservativeResize(npts, Eigen::NoChange);

}

double barycentric(double val1, double val2, double target)
{
    return (target-val1) / (val2-val1);
}

bool crosses(double isoval, double val1, double val2, double &bary)
{
    bary = barycentric(val1, val2, isoval);
    if(bary >= 0 && bary < 1)
        return true;
    return false;
}

int extractIsoline(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXi &faceNeighbors, const Eigen::VectorXd &func, double isoval, Eigen::MatrixXd& isoV, Eigen::MatrixXi& isoE)
{
    int nfaces = F.rows();
    bool *visited = new bool[nfaces];
    for(int i=0; i<nfaces; i++)
        visited[i] = false;

    int ntraces = 0;
    isoV.resize(3 * nfaces, 3);
    isoE.resize(3 * nfaces, 2);

    int npts = 0, nIsos = 0;

    for(int i=0; i<nfaces; i++)
    {
        if(visited[i])
            continue;
        visited[i] = true;
        std::vector<std::vector<Eigen::Vector3d> > traces;
        std::vector<bool> isClose;
        for(int j=0; j<3; j++)
        {
            int vp1 = F(i, (j+1)%3);
            int vp2 = F(i, (j+2)%3);
            double bary;
            if(crosses(isoval, func[vp1], func[vp2], bary))
            {
                std::vector<Eigen::Vector3d> trace;
                trace.push_back( (1.0 - bary) * V.row(vp1) + bary * V.row(vp2) );
                int prevface = i;
                int curface = faceNeighbors(i, j);
                while(curface != -1 && !visited[curface])
                {
                    visited[curface] = true;
                    bool found = false;
                    for(int k=0; k<3; k++)
                    {
                        if(faceNeighbors(curface, k) == prevface)
                            continue;
                        int vp1 = F(curface, (k+1)%3);
                        int vp2 = F(curface, (k+2)%3);
                        double bary;
                        if(crosses(isoval, func[vp1], func[vp2], bary))
                        {
                            trace.push_back( (1.0 - bary) * V.row(vp1) + bary * V.row(vp2) );
                            prevface = curface;
                            curface = faceNeighbors(curface, k);
                            found = true;
                            break;
                        }
                    }
                }
                isClose.push_back(curface != -1);
                traces.push_back(trace);
            }
        }

        std::vector<Eigen::Vector3d> stitchedTraces;
        bool isTraceClose = false;

        if(traces.size() == 1)
        {
            stitchedTraces = traces[0];
            isTraceClose = isClose[0];
        }
        else
        {
            for(int j=traces[1].size()-1; j >= 0; j--)
                stitchedTraces.push_back(traces[1][j]);
            for(int j=0; j<traces[0].size(); j++)
                stitchedTraces.push_back(traces[0][j]);
            isTraceClose = false;
        }

        ntraces++;

        for(int j=0; j<stitchedTraces.size(); j++)
        {
            if(isoV.rows() <= npts)
            {
                isoV.conservativeResize(isoV.rows() + 10000, Eigen::NoChange);
            }
            isoV.row(npts) = stitchedTraces[j].transpose();
            if(isoE.rows() <= nIsos)
            {
                isoE.conservativeResize(isoE.rows() + 10000, Eigen::NoChange);
            }
            if(j < stitchedTraces.size() - 1)
            {
                isoE.row(nIsos) << npts, npts + 1;
                nIsos++;
            }

            else
            {
                if (isTraceClose)
                {
                    isoE.row(nIsos) << npts, npts - (stitchedTraces.size() - 1);
                    nIsos++;
                }
            }
            npts++;

        }
    }
    delete[] visited;
    std::cout << isoE.row(303) << std::endl;
    isoV.conservativeResize(npts, Eigen::NoChange);
    isoE.conservativeResize(nIsos, Eigen::NoChange);
    std::cout << "iso lines extraction done" << std::endl;

    return ntraces;
}

int extractIsoline(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXi &faceNeighbors, const Eigen::VectorXd &func, double isoval, Eigen::MatrixXd& isoV, Eigen::MatrixXi& isoE, Eigen::MatrixXd& extendedV, Eigen::MatrixXi& extendedF)
{
    int nfaces = F.rows();
    bool *visited = new bool[nfaces];
    for(int i=0; i<nfaces; i++)
        visited[i] = false;

    int ntraces = 0;
    isoV.resize(3 * nfaces, 3);
    isoE.resize(3 * nfaces, 2);

    extendedV = V;
    extendedF = F;

    int npts = 0, nIsos = 0;
    int nExtendedVerts = V.rows();
    int nExtendedFaces = F.rows();

    for(int i=0; i<nfaces; i++)
    {
        if(visited[i])
            continue;
        visited[i] = true;
        std::vector<std::vector<Eigen::Vector3d> > traces;
        std::vector<std::vector<std::pair<int, int>>> isoEdgesFaces;
        std::vector<bool> isClose;
        int firstIsoPtsId = -1;
        for(int j=0; j<3; j++)
        {
            int vp1 = F(i, (j+1)%3);
            int vp2 = F(i, (j+2)%3);
            double bary;
            if(crosses(isoval, func[vp1], func[vp2], bary))
            {
                if(firstIsoPtsId == -1)
                    firstIsoPtsId = j;
                std::vector<Eigen::Vector3d> trace;
                std::vector<std::pair<int, int>> isoEdgeFace;
                trace.push_back( (1.0 - bary) * V.row(vp1) + bary * V.row(vp2) );
                int prevface = i;
                int curface = faceNeighbors(i, j);
                while(curface != -1 && !visited[curface])
                {
                    visited[curface] = true;
                    bool found = false;
                    for(int k=0; k<3; k++)
                    {
                        if(faceNeighbors(curface, k) == prevface)
                            continue;
                        int vp1 = F(curface, (k+1)%3);
                        int vp2 = F(curface, (k+2)%3);
                        double bary;
                        if(crosses(isoval, func[vp1], func[vp2], bary))
                        {
                            trace.push_back( (1.0 - bary) * V.row(vp1) + bary * V.row(vp2) );
                            isoEdgeFace.push_back({curface, k});
                            prevface = curface;
                            curface = faceNeighbors(curface, k);
                            found = true;
                            break;
                        }
                    }
                }
                isClose.push_back(curface != -1);
                traces.push_back(trace);

                if(curface != -1)
                {
                    isoEdgeFace.push_back({curface, j});
                }

                isoEdgesFaces.push_back(isoEdgeFace);
            }
        }

        std::vector<Eigen::Vector3d> stitchedTraces;
        bool isTraceClose = false;
        std::vector<std::pair<int, int>> stitchedEdgeFace;

        if(traces.size() == 1)
        {
            stitchedTraces = traces[0];
            isTraceClose = isClose[0];
            stitchedEdgeFace = isoEdgesFaces[0];
        }
        else
        {
            for(int j=traces[1].size()-1; j >= 0; j--)
            {
                stitchedTraces.push_back(traces[1][j]);
                if(traces[1].size() > 1)
                    stitchedEdgeFace.push_back(isoEdgesFaces[1][j]);
            }
            stitchedEdgeFace.push_back({i, firstIsoPtsId});
            for(int j=0; j<traces[0].size(); j++)
            {
                stitchedTraces.push_back(traces[0][j]);
                if(traces[0].size() > 1)
                    stitchedEdgeFace.push_back(isoEdgesFaces[0][j]);
            }

            isTraceClose = false;
        }

        ntraces++;

        for(int j=0; j<stitchedTraces.size(); j++)
        {
            if(isoV.rows() <= npts)
            {
                isoV.conservativeResize(isoV.rows() + 10000, Eigen::NoChange);
            }
            isoV.row(npts) = stitchedTraces[j].transpose();


            if(isoE.rows() <= nIsos)
            {
                isoE.conservativeResize(isoE.rows() + 10000, Eigen::NoChange);
            }
            if(j < stitchedTraces.size() - 1)
            {
                isoE.row(nIsos) << npts, npts + 1;
                nIsos++;
            }
            else
            {
                if (isTraceClose)
                {
                    isoE.row(nIsos) << npts, npts - (stitchedTraces.size() - 1);
                    nIsos++;
                }
            }

            if(extendedV.rows() <= nExtendedVerts)
            {
                extendedV.conservativeResize(extendedV.rows() + 10000, Eigen::NoChange);
            }
            extendedV.row(nExtendedVerts) = stitchedTraces[j].transpose();

            if(extendedF.rows() <= nExtendedFaces + 1)
            {
                extendedF.conservativeResize(extendedF.rows() + 10000, Eigen::NoChange);
            }

            if(j < stitchedTraces.size() - 1)
            {
                // old face
                extendedF(stitchedEdgeFace[j].first, stitchedEdgeFace[j].second) = nExtendedVerts;
                extendedF(stitchedEdgeFace[j].first, (stitchedEdgeFace[j].second + 1) % 3) = nExtendedVerts + 1;

                // two new faces
                extendedF.row(nExtendedFaces) << F(stitchedEdgeFace[j].first, stitchedEdgeFace[j].second), nExtendedVerts + 1, nExtendedVerts;
                extendedF.row(nExtendedFaces + 1) << F(stitchedEdgeFace[j].first, stitchedEdgeFace[j].second), F(stitchedEdgeFace[j].first, (stitchedEdgeFace[j].second + 1) % 3), nExtendedVerts + 1;

                nExtendedFaces += 2;
            }
            else
            {
                if (isTraceClose)
                {
                    extendedF(stitchedEdgeFace[j].first, stitchedEdgeFace[j].second) = nExtendedVerts;
                    extendedF(stitchedEdgeFace[j].first, (stitchedEdgeFace[j].second + 1) % 3) = nExtendedVerts - (stitchedTraces.size() - 1);

                    // two new faces
                    extendedF.row(nExtendedFaces) << F(stitchedEdgeFace[j].first, stitchedEdgeFace[j].second), nExtendedVerts - (stitchedTraces.size() - 1), nExtendedVerts;
                    extendedF.row(nExtendedFaces + 1) << F(stitchedEdgeFace[j].first,stitchedEdgeFace[j].second), F(stitchedEdgeFace[j].first, (stitchedEdgeFace[j].second + 1) % 3), nExtendedVerts - (stitchedTraces.size() - 1);

                    nExtendedFaces += 2;
                }
            }

            npts++;
            nExtendedVerts++;

        }

    }
    delete[] visited;
    isoV.conservativeResize(npts, Eigen::NoChange);
    isoE.conservativeResize(nIsos, Eigen::NoChange);
    extendedV.conservativeResize(nExtendedVerts, Eigen::NoChange);
    extendedF.conservativeResize(nExtendedFaces, Eigen::NoChange);

    return ntraces;
}