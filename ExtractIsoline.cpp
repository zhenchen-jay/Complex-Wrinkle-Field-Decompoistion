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
    assert(z.rows() == nVerts &&
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

        if(traces.size() == 1)
        {
            ntraces++;

            for(int j=0; j<traces[0].size(); j++)
            {
                if(isoV.rows() <= npts)
                {
                    isoV.conservativeResize(isoV.rows() + 10000, Eigen::NoChange);
                }
                isoV.row(npts) = traces[0][j].transpose();
                if(isoE.rows() <= nIsos)
                {
                    isoE.conservativeResize(isoE.rows() + 10000, Eigen::NoChange);
                }
                if(j < traces[0].size() - 1)
                {
                    isoE.row(nIsos) << npts, npts + 1;
                    nIsos++;
                }

                else
                {
                    if (isClose[0])
                    {
                        isoE.row(nIsos) << npts, npts - (traces[0].size() - 1);
                        nIsos++;
                    }
                }
                npts++;

            }


        }
        if(traces.size() == 2)
        {
            ntraces++;
//            int nterms = traces[0].size() + traces[1].size();
//            std::cout << nterms << " 0 0" << std::endl;
            for(int j=traces[1].size()-1; j >= 0; j--)
            {
                if(isoV.rows() <= npts)
                {
                    isoV.conservativeResize(isoV.rows() + 10000, Eigen::NoChange);
                }
                isoV.row(npts) = traces[1][j].transpose();

                if(isoE.rows() <= nIsos)
                {
                    isoE.conservativeResize(isoE.rows() + 10000, Eigen::NoChange);
                }
                if(j > 0)
                {
                    isoE.row(nIsos) << npts, npts + 1;
                    nIsos++;
                }
                else
                {
                    if(isClose[1])
                    {
                        isoE.row(nIsos) << npts, npts - (traces[1].size() - 1);
                        nIsos++;
                    }
                }
                npts++;
            }
//                std::cout << traces[1][j].transpose() << std::endl;
            for(int j=0; j<traces[0].size(); j++)
            {
                if(isoV.rows() <= npts)
                {
                    isoV.conservativeResize(isoV.rows() + 10000, Eigen::NoChange);
                }
                isoV.row(npts) = traces[0][j].transpose();

                if(isoE.rows() <= nIsos)
                {
                    isoE.conservativeResize(isoE.rows() + 10000, Eigen::NoChange);
                }
                if(j < traces[0].size() - 1)
                {
                    isoE.row(nIsos) << npts, npts + 1;
                    nIsos++;
                }

                else
                {
                    if(isClose[0])
                    {
                        isoE.row(nIsos) << npts, npts - (traces[0].size() - 1);
                        nIsos++;
                    }
                }
                npts++;

            }
//                std::cout << traces[0][j].transpose() << std::endl;
        }
    }
    delete[] visited;
    std::cout << isoE.row(303) << std::endl;
    isoV.conservativeResize(npts, Eigen::NoChange);
    isoE.conservativeResize(nIsos, Eigen::NoChange);
    std::cout << "iso lines extraction done" << std::endl;

    return ntraces;
}