#include "Subdivision.h"

#include <iomanip>

#include "BaseLoop.h"
#include "ComplexLoop.h"

namespace ComplexWrinkleField {
    // Standard Loop subdivision
    void Subdivide(const Mesh& mesh,                    // input mesh
                   Mesh& upmesh,                        // upsampled mesh
                   int level,                           // upsampling level
                   bool fixedBnd,                       // fix boundary points
                   SparseMatrixX* upS0                  // upsampling matrix
    ) {
        std::unique_ptr<BaseLoop> loopPtr = std::make_unique<BaseLoop>();
        int nverts = mesh.GetVertCount();

        MatrixX X;
        mesh.GetPos(X);

        SparseMatrixX S0;

        if (upS0)
        {
            S0.resize(nverts, nverts);
            S0.setIdentity();
        }

        upmesh = mesh;
        loopPtr->SetBndFixFlag(fixedBnd);
        for (int l = 0; l < level; ++l)
        {
            loopPtr->SetMesh(upmesh);
            SparseMatrixX tmpS0, tmpS1;
            ComplexSparseMatrixX tmpCS0;
            loopPtr->BuildS0(tmpS0);

            X = tmpS0 * X;

            std::vector<Vector3> points;
            ConvertToVector3(X, points);

            std::vector< std::vector<int> > edgeToVert;
            loopPtr->GetSubdividedEdges(edgeToVert);

            std::vector< std::vector<int> > faceToVert;
            loopPtr->GetSubdividedFaces(faceToVert);

            upmesh.Populate(points, faceToVert, edgeToVert);

            if (upS0)
                S0 = tmpS0 * S0;
        }
        if (upS0)
            *upS0 = S0;
    }

    // CWF Loop subdivision
    void Subdivide(const CWF& cwf,								    // input CWF
                   CWF& upcwf,									    // output CWF
                   int level,                                       // upsampling level
                   bool fixedBnd,                                   // fix boundary points
                   SparseMatrixX* upS0,					            // upsampling matrix
                   SparseMatrixX* upS1,					            // upsampling matrix
                   ComplexSparseMatrixX* upComplexS0 ) 	            // upsampling matrix
    {
        std::unique_ptr<BaseLoop> loopPtr = std::make_unique<ComplexLoop>();
        int nverts = cwf._mesh.GetVertCount();
        int nedges = cwf._mesh.GetEdgeCount();

        MatrixX X;
        cwf._mesh.GetPos(X);

        SparseMatrixX S0, S1;
        ComplexSparseMatrixX CS0;

        if (upS0)
        {
            S0.resize(nverts, nverts);
            S0.setIdentity();
        }

        if (upS1)
        {
            S1.resize(nedges, nedges);
            S1.setIdentity();
        }

        if (upComplexS0)
        {
            CS0.resize(nverts, nverts);
            CS0.setIdentity();
        }

        upcwf = cwf;
        loopPtr->SetBndFixFlag(fixedBnd);

        for (int l = 0; l < level; ++l)
        {
            loopPtr->SetMesh(upcwf._mesh);
            SparseMatrixX tmpS0, tmpS1;
            ComplexSparseMatrixX tmpCS0;
            loopPtr->BuildS0(tmpS0);
            loopPtr->BuildS1(tmpS1);
            loopPtr->BuildComplexS0(upcwf._omega, tmpCS0);

            X = tmpS0 * X;
            upcwf._amp = tmpS0 * upcwf._amp;
            upcwf._omega = tmpS1 * upcwf._omega;
            upcwf._zvals = tmpCS0 * upcwf._zvals;

            std::vector<Vector3> points;
            ConvertToVector3(X, points);

            std::vector< std::vector<int> > edgeToVert;
            loopPtr->GetSubdividedEdges(edgeToVert);

            std::vector< std::vector<int> > faceToVert;
            loopPtr->GetSubdividedFaces(faceToVert);

            upcwf._mesh.Populate(points, faceToVert, edgeToVert);

            if (upS0)
                S0 = tmpS0 * S0;
            if (upS1)
                S1 = tmpS1 * S1;
            if (upComplexS0)
                CS0 = tmpCS0 * CS0;
        }
        if (upS0)
            *upS0 = S0;
        if (upS1)
            *upS1 = S1;
        if (upComplexS0)
            *upComplexS0 = CS0;
    }
}