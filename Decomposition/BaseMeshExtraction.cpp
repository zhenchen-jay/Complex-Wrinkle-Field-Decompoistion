#include "BaseMeshExtraction.h"
#include <igl/principal_curvature.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>

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
    Eigen::VectorXd doubleArea, vertAreaInv;
    igl::doublearea(wrinkledPos, wrinkledFaces, doubleArea);
    vertAreaInv.setZero(wrinkledPos.rows());
    for(int i = 0; i < doubleArea.rows(); i++)
    {
        for(int j = 0; j < 3; j++)
            vertAreaInv[wrinkledFaces(i, j)] += doubleArea[i] / 6.0;
    }
    SparseMatrixX massInv(vertAreaInv.rows(), vertAreaInv.rows());
    std::vector<TripletX> T;
    for(int i = 0; i < vertAreaInv.rows(); i++)
        T.push_back({i, i, vertAreaInv[i]});
    massInv.setFromTriplets(T.begin(), T.end());

    // step 1: compute mean curvatures
    Eigen::MatrixXd PD1, PD2;
    Eigen::VectorXd PV1, PV2;
    igl::principal_curvature(wrinkledPos, wrinkledFaces, PD1, PD2, PV1, PV2);
    Eigen::VectorXd H = (PV1 + PV2) / 2;

    // step 2: extract zero iso-pts and iso-lines
    Eigen::MatrixXd isolinePos;
    Eigen::MatrixXi isolineEdges;
    extractIsoline(wrinkledPos, wrinkledFaces, wrinkledFaceNeighbors, H, 0, isolinePos, isolineEdges);

    // step 3: update the wrinkle pos
    basemeshPos = wrinkledPos;
    basemeshFaces = wrinkledFaces;
    SparseMatrixX L;
    igl::cotmatrix(wrinkledPos, wrinkledFaces, L);

    // step 4: build bilaplacian (quadratic bending)
    SparseMatrixX BiLap = L * massInv * L, fullBiLap(3 * wrinkledPos.rows(), 3 * wrinkledPos.rows());
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



}