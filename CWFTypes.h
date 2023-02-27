#pragma once

#include "MeshLib/types.h"
#include "MeshLib/Mesh.h"

class CWF
{
public:
	CWF();
	CWF(const VectorX& amp, const VectorX& omega, const ComplexVectorX& zvals, const Mesh& mesh);

	void initialization(const VectorX& amp, const VectorX& omega, const ComplexVectorX& zvals, const Mesh& mesh);

	VectorX _amp;				// wrinkle amplitude, stored as per vertex scalar
	VectorX _omega;				// wrinkle frequency, stored as per edge one form
	ComplexVectorX _zvals;		// wrinkle phase, stored as per vertex complex
	Mesh _mesh;					// base mesh geometry
};