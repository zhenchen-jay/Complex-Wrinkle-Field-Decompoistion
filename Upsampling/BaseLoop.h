#pragma once
#include "../MeshLib/Mesh.h"
#include "../CWFTypes.h"

class BaseLoop	// We modify the Loop.h
{
protected:
	Mesh _mesh;
	bool _isFixBnd;

public:
	BaseLoop() : _isFixBnd(false) { }

	virtual ~BaseLoop() = default;

	inline Mesh GetMesh() const { return _mesh; }
	inline void  SetMesh(const Mesh& mesh) { _mesh = mesh; }
	inline void SetBndFixFlag(const bool& isFixBnd) { _isFixBnd = isFixBnd; }

	void BuildS0(SparseMatrixX& A) const;
	void BuildS1(SparseMatrixX& A) const;
    void virtual BuildComplexS0(const VectorX& omega, ComplexSparseMatrixX& A) = 0;
   
	void GetSubdividedEdges(std::vector< std::vector<int> >& edgeToVert) const;
	void GetSubdividedFaces(std::vector< std::vector<int> >& faceToVert);

	bool IsVertRegular(int vert) const;
	bool AreIrregularVertsIsolated() const;

	void virtual CWFSubdivide(
		const CWF& cwf,								// input CWF
		CWF &upcwf,									// output CWF
		int level, 
		SparseMatrixX* upS0 = NULL,					// upsampled matrix for scalar
		SparseMatrixX* upS1 = NULL,					// upsampled matrix for one form
		ComplexSparseMatrixX* upComplexS0 = NULL	// upsampled matrix for complex scalar
	) = 0;
    Mesh meshSubdivide(int level);

protected:
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