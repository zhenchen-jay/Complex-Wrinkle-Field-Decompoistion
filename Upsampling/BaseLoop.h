#pragma once
#include <iomanip>
#include "../MeshLib/Mesh.h"
#include "../CWFTypes.h"

namespace ComplexWrinkleField {
    class BaseLoop {
    public:
        BaseLoop() : _mesh(nullptr), _isFixBnd(false) { }

        virtual ~BaseLoop() = default;

        inline Mesh GetMesh() const { return *_mesh; }
        inline void  SetMesh(const Mesh& mesh) { _mesh = &mesh; }
        inline void SetBndFixFlag(const bool& isFixBnd) { _isFixBnd = isFixBnd; }

        void BuildS0(SparseMatrixX& A) const;
        void BuildS1(SparseMatrixX& A) const;
        void virtual BuildComplexS0(const VectorX& omega, ComplexSparseMatrixX& A) const {
            assert("This shouldn't be called");
        }

        void GetSubdividedEdges(std::vector< std::vector<int> >& edgeToVert) const;
        void GetSubdividedFaces(std::vector< std::vector<int> >& faceToVert) const;

        bool IsVertRegular(int vert) const;
        bool AreIrregularVertsIsolated() const;

    protected:
        Mesh const* _mesh;
        bool _isFixBnd;

        int _GetVertVertIndex(int vert) const;
        int _GetEdgeVertIndex(int edge) const;
        int _GetFaceVertIndex(int face) const;

        int _GetEdgeEdgeIndex(int edge, int vertInEdge) const;
        int _GetFaceEdgeIndex(int face, int edgeInFace) const;

        int _GetCentralFaceIndex(int face) const;
        int _GetCornerFaceIndex(int face, int vertInFace) const;

        Scalar _GetAlpha(int vert) const;
        Scalar _GetBeta(int vert) const;

        void _AssembleVertEvenInterior(int vi, TripletInserter out) const;
        void _AssembleVertEvenBoundary(int vi, TripletInserter out) const;
        void _AssembleVertOddInterior(int edge, TripletInserter out) const;
        void _AssembleVertOddBoundary(int edge, TripletInserter out) const;

        void _AssembleEdgeEvenInterior(int edge, int vertInEdge, TripletInserter out) const;
        void _AssembleEdgeEvenBoundary(int edge, int vertInEdge, TripletInserter out) const;
        void _AssembleEdgeEvenPartialBoundary(int edge, int vertInEdge, TripletInserter out) const;
        void _AssembleEdgeOdd(int face, int edgeInFace, TripletInserter out) const;

        void _InsertEdgeEdgeValue(int row, int col, int vert, int rSign, Scalar val, TripletInserter out) const;
        void _InsertEdgeFaceValue(int row, int face, int vert, int rSign, Scalar val, TripletInserter out) const;
    };
}

