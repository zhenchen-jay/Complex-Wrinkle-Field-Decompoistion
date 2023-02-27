#include "CWFTypes.h"

CWF::CWF()
{

}

CWF::CWF(const VectorX& amp, const VectorX& omega, const ComplexVectorX& zvals, const Mesh& mesh)
{
	initialization(amp, omega, zvals, mesh);
}

void CWF::initialization(const VectorX& amp, const VectorX& omega, const ComplexVectorX& zvals, const Mesh& mesh)
{
	_amp = amp;
	_omega = omega;
	_zvals = zvals;
	_mesh = mesh;
}