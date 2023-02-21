#include "CWFDecomposition.h"
#include "../LoadSaveIO.h"
#include <igl/per_vertex_normals.h>

void CWFDecomposition::intialization(const Mesh &baseMesh, const std::vector<std::complex<double>> &unitZvals,
									 const Eigen::VectorXd &amp, const Eigen::VectorXd &omega, int upsampleTimes)
{
	_baseMesh = baseMesh;
	_unitZvals = unitZvals;
	_amp = amp;
	_omega = omega;
	_upsampleTimes = upsampleTimes;
	_subOp = std::make_shared<ComplexLoop>();
	_subOp->SetMesh(_baseMesh);
	_upMesh = _subOp->meshSubdivide(upsampleTimes);
	_upMesh.GetPos(_upV);
	_upMesh.GetFace(_upF);
	igl::per_vertex_normals(_upV, _upF, _upN);
}

void CWFDecomposition::getCWF(Mesh &baseMesh, std::vector<std::complex<double>> &unitZvals, Eigen::VectorXd &amp,
							  Eigen::VectorXd &omega)
{
	baseMesh = _baseMesh;
	unitZvals = _unitZvals;
	amp = _amp;
	omega = _omega;
}

void CWFDecomposition::convertCWF2Variables(const std::vector<std::complex<double>>& unitZvals, const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, Eigen::VectorXd& x)
{
	int nverts = _baseMesh.GetVertCount();
	int nedges = _baseMesh.GetEdgeCount();

	x.setZero(3 * nverts + nedges);
	for (int i = 0; i < nverts; i++)
	{
		x[3 * i] = amp[i];
		x[3 * i + 1] = unitZvals[i].real();
		x[3 * i + 2] = unitZvals[i].imag();
	}

	for (int i = 0; i < nedges; i++)
	{
		x[i + 3 * nverts] = omega[i];
	}
}

void CWFDecomposition::convertVariables2CWF(const Eigen::VectorXd& x, std::vector<std::complex<double>>& unitZvals, Eigen::VectorXd& amp, Eigen::VectorXd& omega)
{
	int nverts = _baseMesh.GetVertCount();
	int nedges = _baseMesh.GetEdgeCount();

	if (omega.rows() != nedges)
		omega.setZero(nedges);
	if (amp.rows() != nverts)
		amp.setZero(nverts);
	if (unitZvals.size() != nverts)
		unitZvals.resize(nverts);

	for (int i = 0; i < nverts; i++)
	{
		amp[i] = x[3 * i];
		unitZvals[i] = std::complex<double>(x[3 * i + 1], x[3 * i + 2]);
	}

	for (int i = 0; i < nedges; i++)
	{
		omega[i] = x[i + 3 * nverts];
	}
}

void CWFDecomposition::convertCWF2Variables(Eigen::VectorXd &x)
{
	convertCWF2Variables(_unitZvals, _amp, _omega, x);
}

void CWFDecomposition::convertVariables2CWF(const Eigen::VectorXd &x)
{
	convertVariables2CWF(x, _unitZvals, _amp, _omega);
}

double CWFDecomposition::computeDifferenceEnergy(const Eigen::VectorXd &x, Eigen::VectorXd *grad,
												 Eigen::SparseMatrix<double> *hess)
{
	auto energyComputation = [&](const Eigen::VectorXd& x)
	{
		std::vector<std::complex<double>> zvals, upZvals, unitZvals;
		Eigen::VectorXd amp, omega;
		convertVariables2CWF(x, unitZvals, amp, omega);

		rescaleZvals(unitZvals, amp, zvals);
		Eigen::VectorXd upOmega;

		std::shared_ptr<BaseLoop> subOp;
		subOp = std::make_shared<ComplexLoop>();
		subOp->SetMesh(_baseMesh);

		subOp->CWFSubdivide(omega, zvals, upOmega, upZvals, _upsampleTimes);
		Eigen::MatrixXd wrinkledPos = _upV;
		double energy = 0;
		for(int i = 0; i < _upV.rows(); i++)
		{
			wrinkledPos.row(i) += upZvals[i].real() * _upN.row(i);
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

		//for(int i = 0; i < nvars; i++)
		//{
		//	double eps = 1e-6;
		//	auto x1 = x, x2 = x;
		//	x1[i] = x[i] + eps;
		//	x2[i] = x[i] - eps;

		//	double energy1 = energyComputation(x1);
		//	double energy2 = energyComputation(x2);

		//	double val = (energy1 - energy2) / (2 * eps);   // finite difference
		//	(*grad)[i] = val;
		//}

	}

	return energy;

}

void CWFDecomposition::optimizeCWF()
{
	Eigen::VectorXd x0;
	convertCWF2Variables(x0);

	int iter = 0;
	for(; iter < 10000; iter++)
	{
		// gradient descent
		Eigen::VectorXd grad;
		double f0 = computeDifferenceEnergy(x0, &grad, NULL);

		if (grad.norm() < 1e-6)
		{
			std::cout << "small gradient norm: " << grad.norm() << ", current energy: " << f0 << std::endl;
			break;
		}
		
		double alpha = 1;
		Eigen::VectorXd x1;
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
                std::vector<std::complex<double>> zvals;
                Eigen::VectorXd omega;
                Eigen::MatrixXi baseF;
                _baseMesh.GetFace(baseF);
                omega = swapEdgeVec(baseF, _omega, 1);
                rescaleZvals(_unitZvals, _amp, zvals);
                saveVertexAmp("tmpAmp_" + std::to_string(iter / 100) + ".txt", _amp);
                saveEdgeOmega("tmpOmega_" + std::to_string(iter / 100) + ".txt", omega);
                saveVertexZvals("tmpUnitZval_" + std::to_string(iter / 100) + ".txt", _unitZvals);
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