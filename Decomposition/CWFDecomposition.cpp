#include "CWFDecomposition.h"
#include "../LoadSaveIO.h"
#include <igl/per_vertex_normals.h>

void CWFDecomposition::intialization(const CWF& cwf, int upsampleTimes)
{
	_baseCWF = cwf;
	_upsampleTimes = upsampleTimes;
	_subOp = std::make_shared<ComplexLoop>();
	_subOp->SetMesh(_baseCWF._mesh);
	_upMesh = _subOp->meshSubdivide(upsampleTimes);
	_upMesh.GetPos(_upV);
	_upMesh.GetFace(_upF);
	igl::per_vertex_normals(_upV, _upF, _upN);
}

void CWFDecomposition::getCWF(CWF& cwf)
{
	cwf = _baseCWF;
}

void CWFDecomposition::convertCWF2Variables(const CWF& cwf, VectorX& x)
{
	int nverts = _baseCWF._mesh.GetVertCount();
	int nedges = _baseCWF._mesh.GetEdgeCount();

	x.setZero(3 * nverts + nedges);
	for (int i = 0; i < nverts; i++)
	{
		x[3 * i] = cwf._amp[i];
		x[3 * i + 1] = cwf._zvals[i].real();
		x[3 * i + 2] = cwf._zvals[i].imag();
	}

	for (int i = 0; i < nedges; i++)
	{
		x[i + 3 * nverts] = cwf._omega[i];
	}
}

void CWFDecomposition::convertVariables2CWF(const VectorX& x, CWF& cwf)
{
	int nverts = _baseCWF._mesh.GetVertCount();
	int nedges = _baseCWF._mesh.GetEdgeCount();

	if (cwf._omega.rows() != nedges)
		cwf._omega.setZero(nedges);
	if (cwf._amp.rows() != nverts)
		cwf._amp.setZero(nverts);
	if (cwf._zvals.size() != nverts)
		cwf._zvals.resize(nverts);

	for (int i = 0; i < nverts; i++)
	{
		cwf._amp[i] = x[3 * i];
		cwf._zvals[i] = std::complex<double>(x[3 * i + 1], x[3 * i + 2]);
	}

	for (int i = 0; i < nedges; i++)
	{
		cwf._omega[i] = x[i + 3 * nverts];
	}
}

void CWFDecomposition::convertCWF2Variables(VectorX &x)
{
	convertCWF2Variables(_baseCWF, x);
}

void CWFDecomposition::convertVariables2CWF(const VectorX &x)
{
	convertVariables2CWF(x, _baseCWF);
}

double CWFDecomposition::computeDifferenceEnergy(const VectorX &x, VectorX *grad, SparseMatrixX *hess)
{
	auto energyComputation = [&](const VectorX& x)
	{
		std::shared_ptr<BaseLoop> subOp;
		subOp = std::make_shared<ComplexLoop>();
		CWF upcwf, basecwf;
		convertVariables2CWF(x, basecwf);
		basecwf._mesh = _baseCWF._mesh;
		subOp->CWFSubdivide(basecwf, upcwf, _upsampleTimes);
		Eigen::MatrixXd wrinkledPos = _upV;
		double energy = 0;
		for(int i = 0; i < _upV.rows(); i++)
		{
			wrinkledPos.row(i) += upcwf._amp[i] * upcwf._zvals[i].real() * _upN.row(i);
			energy += 0.5 * (wrinkledPos.row(i) - _wrinkledV.row(i)).squaredNorm();
		}
		return energy;
	};
	double energy = energyComputation(x);

	if(grad)
	{
		// finite difference
		int nvars = x.rows();
		grad->setZero(nvars);

		tbb::parallel_for(
			tbb::blocked_range<int>(0u, nvars),
			[&](const tbb::blocked_range<int>& range)
			{
				for (int i = range.begin(); i != range.end(); ++i)
				{
					double eps = 1e-6;
					auto x1 = x, x2 = x;
					x1[i] = x[i] + eps;
					x2[i] = x[i] - eps;

					double energy1 = energyComputation(x1);
					double energy2 = energyComputation(x2);

					double val = (energy1 - energy2) / (2 * eps);   // finite difference
					(*grad)[i] = val;
				}
			}
		);

	}

	return energy;

}

void CWFDecomposition::optimizeCWF()
{
	VectorX x0;
	convertCWF2Variables(x0);

	int iter = 0;
	for(; iter < 10000; iter++)
	{
		// gradient descent
		VectorX grad;
		double f0 = computeDifferenceEnergy(x0, &grad, NULL);

		if (grad.norm() < 1e-6)
		{
			std::cout << "small gradient norm: " << grad.norm() << ", current energy: " << f0 << std::endl;
			break;
		}
		
		double alpha = 1;
		VectorX x1;
		double f1 = f0;

		bool energyDecrease = false;
		// line search, probably Armijo line search is better. But I don't trust the finite difference 
		while (alpha > 1e-6)
		{
			x1 = x0 - alpha * grad;
			f1 = computeDifferenceEnergy(x1, NULL, NULL);
			if (f1 < f0)
			{
				energyDecrease = true;
				break;
			}
			alpha *= 0.5;
		}
		if (!energyDecrease)
		{
			std::cout << "not a descent direction!" << std::endl;
			break;
		}
		else
		{
            std::cout << "iter: " << iter << ", f0: " << f0 << ", f1: " << f1 << ", line search rate: " << alpha << std::endl;
            std::cout << "var update: " << (x1 - x0).norm() << ", fval update: " << f0 - f1 << ", grad norm: " << grad.norm() << std::endl;
			if (f0 - f1 < 1e-10)
			{
				std::cout << "small energy update: " << f0 - f1 << std::endl;
				break;
			}
			if ((x1 - x0).norm() < 1e-10)
			{
				std::cout << "small position update: " << (x1 - x0).norm() << std::endl;
				break;
			}
			std::swap(x0, x1);

            if(iter % 20 == 0)
            {
                convertVariables2CWF(x0);
                ComplexVectorX zvals;
                VectorX omega;
                Eigen::MatrixXi baseF;
                _baseCWF._mesh.GetFace(baseF);
                omega = swapEdgeVec(baseF, _baseCWF._omega, 1);
                rescaleZvals(_baseCWF._zvals, _baseCWF._amp, zvals);
                saveVertexAmp("tmpAmp_" + std::to_string(iter / 100) + ".txt", _baseCWF._amp);
                saveEdgeOmega("tmpOmega_" + std::to_string(iter / 100) + ".txt", omega);
                saveVertexZvals("tmpUnitZval_" + std::to_string(iter / 100) + ".txt", _baseCWF._zvals);
                saveVertexZvals("tmpZval_" + std::to_string(iter / 100) + ".txt", zvals);
            }
		}
	}

	if (iter >= 10000)
	{
		std::cout << "reach the maximum iterations: " << iter << std::endl;
	}

	convertVariables2CWF(x0);

}