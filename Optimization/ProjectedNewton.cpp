#include "Newton.h"
#include "ProjectedNewton.h"
#include <iostream>
#include <Eigen/CholmodSupport>
#include "../Timer.h"

static void IdentifyBindingSet(const Eigen::VectorXd& x, const Eigen::VectorXd& dir, const Eigen::VectorXd& lx, const Eigen::VectorXd& ux, Eigen::VectorXi& flag, double tol = 1e-7)
{
	const int DIM = x.rows();
	if (flag.size() != DIM)
	{
		std::cout << "flag and x size do not match, reinitialize flag size" << std::endl;
		flag.setZero(DIM);
	}

	for (int i = 0; i < DIM; i++)
	{
		if (lx[i] != std::numeric_limits<double>::min())
		{
			if (x[i] - lx[i] < tol && dir[i] < 0)		// x + dir will violate the lowerbound constraint
				flag[i] = 1;
		}
		if (ux[i] != std::numeric_limits<double>::max())
		{
			if (ux[i] - x[i] < tol && dir[i] > 0)		// x + dir will violate the upperbound constriant
				flag[i] = 1;
		}
	}
}

void ProjectedNewtonSolver(std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc, std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> findMaxStep, const Eigen::VectorXd& lx, const Eigen::VectorXd& ux, Eigen::VectorXd& x0, int numIter, double gradTol, double xTol, double fTol, bool displayInfo)
{
	const int DIM = x0.rows();

	int nconstraints = 0;
	for (int i = 0; i < DIM; i++)
	{
		if (lx[i] != std::numeric_limits<double>::min())
			nconstraints++;
		if (ux[i] != std::numeric_limits<double>::max())
			nconstraints++;
	}
	if (!nconstraints)	// no constraints, just do newton method
		return NewtonSolver(objFunc, findMaxStep, x0, numIter, gradTol, xTol, fTol, displayInfo);

	auto bbxProj = [&](const Eigen::VectorXd& x)
	{
		Eigen::VectorXd projx = x;
		for (int i = 0; i < DIM; i++)
		{
			if (lx[i] == std::numeric_limits<double>::min() || ux[i] == std::numeric_limits<double>::max())
				continue;

			if (projx[i] < lx[i])
				projx[i] = lx[i];
			if (projx[i] > ux[i])
				projx[i] = ux[i];
		}
		return projx;
	};

	auto freeDOFsProj = [&](const Eigen::VectorXd& dir, const Eigen::VectorXi& bindingSetFlags)
	{
		Eigen::VectorXd projDir = dir;
		for (int i = 0; i < DIM; i++)
		{
			if (bindingSetFlags[i])	// fixed (binding) variable
				projDir[i] = 0;
		}
		return projDir;
	};

	auto projGradNorm = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& grad, double tol = 1e-7)		// the KKT residual
	{
		double projNorm = 0;
		for (int i = 0; i < DIM; i++)
		{
			bool isFree = true;
			if (lx[i] != std::numeric_limits<double>::min())
			{
				if (x[i] - lx[i] < tol && grad[i] > 0)		
					isFree = false;
			}
			if (ux[i] != std::numeric_limits<double>::max())
			{
				if (ux[i] - x[i] < tol && grad[i] < 0)		
					isFree = false;
			}

			if (isFree)
			{
				projNorm += grad[i] * grad[i];
			}
		}
		return std::sqrt(projNorm);
	};


	Eigen::VectorXd grad = Eigen::VectorXd::Zero(DIM);
	Eigen::SparseMatrix<double> hessian;

	Eigen::VectorXd neggrad, delta_x;
	double maxStepSize = 1.0;
	double reg = 1e-8;

	bool isProj = true;
	Timer<std::chrono::high_resolution_clock> totalTimer;
	double totalAssemblingTime = 0;
	double totalSolvingTime = 0;
	double totalLineSearchTime = 0;

	totalTimer.start();
	int i = 0;

	auto x1 = x0;
	x0 = bbxProj(x0);

	double f = objFunc(x0, NULL, NULL, false);
	if (f == 0)
	{
		std::cout << "energy = 0, return" << std::endl;
	}

	bool isSmallPerturbNeeded = false;

	for (; i < numIter; i++)
	{
		if (displayInfo)
			std::cout << "\niter: " << i << std::endl;
		Timer<std::chrono::high_resolution_clock> localTimer;
		localTimer.start();
		double f = objFunc(x0, &grad, &hessian, isProj);
		localTimer.stop();
		double localAssTime = localTimer.elapsed<std::chrono::milliseconds>() * 1e-3;
		totalAssemblingTime += localAssTime;

		localTimer.start();
		Eigen::SparseMatrix<double> H = hessian;
		Eigen::SparseMatrix<double> I(DIM, DIM);
		I.setIdentity();
		std::cout << "num of nonzeros: " << H.nonZeros() << ", rows: " << H.rows() << ", cols: " << H.cols() << std::endl;

		if (isSmallPerturbNeeded && isProj)
			H += reg * I;

		Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver(H);

		//Eigen::CholmodSimplicialLLT<Eigen::SparseMatrix<double> > solver(H);


		while (solver.info() != Eigen::Success)
		{
			if (displayInfo)
			{
				if (isProj) {
					std::cout << "some small perturb is needed to remove round-off error, current reg = " << reg << std::endl;
				}

				else
					std::cout << "Matrix is not positive definite, current reg = " << reg << std::endl;
			}

			if (isProj)
				isSmallPerturbNeeded = true;

			H = hessian + reg * I;
			solver.compute(H);
			reg = std::max(2 * reg, 1e-16);

			if (reg > 1e4)
			{
				std::cout << "reg is too large, use SPD hessian instead." << std::endl;
				reg = 1e-6;
				isProj = true;
				f = objFunc(x0, &grad, &hessian, isProj);
			}
		}

		neggrad = -grad;
		delta_x = solver.solve(neggrad);

		localTimer.stop();
		double localSolvingTime = localTimer.elapsed<std::chrono::milliseconds>() * 1e-3;
		totalSolvingTime += localSolvingTime;

		// build binding set
		Eigen::VectorXi bindingsetFlag(DIM);
		bindingsetFlag.setZero();

		IdentifyBindingSet(x0, neggrad, lx, ux, bindingsetFlag, gradTol / std::sqrt(DIM));
		IdentifyBindingSet(x0, delta_x, lx, ux, bindingsetFlag, gradTol / std::sqrt(DIM));

		// project the direction
		delta_x = freeDOFsProj(delta_x, bindingsetFlag);

		maxStepSize = findMaxStep(x0, delta_x);

		localTimer.start();
		double rate = ArmijoLineSearch(x0, grad, delta_x, objFunc, maxStepSize, bbxProj);
		localTimer.stop();
		double localLinesearchTime = localTimer.elapsed<std::chrono::milliseconds>() * 1e-3;
		totalLineSearchTime += localLinesearchTime;


		if (!isProj)
		{
			reg *= 0.5;
			reg = std::max(reg, 1e-16);
		}
		else
			reg = 1e-8;

		x0 = x0 + rate * delta_x;
		x0 = bbxProj(x0);

		double fnew = objFunc(x0, &grad, NULL, isProj);
		if (displayInfo)
		{
			std::cout << "line search rate : " << rate << ", actual hessian : " << !isProj << ", reg = " << reg << std::endl;
			std::cout << "f_old: " << f << ", f_new: " << fnew << ", grad norm: " << grad.norm() << ", delta x: " << rate * delta_x.norm() << ", delta_f: " << f - fnew << std::endl;
			std::cout << "timing info (in total seconds): " << std::endl;
			std::cout << "assembling took: " << totalAssemblingTime << ", LLT solver took: " << totalSolvingTime << ", line search took: " << totalLineSearchTime << std::endl;
		}


		if ((f - fnew) / f < 1e-5 || delta_x.norm() < 1e-5 || projGradNorm(x0, grad, gradTol / std::sqrt(DIM)) < 1e-4)
		{
			isProj = false;
		}


		if (rate < 1e-8)
		{
			std::cout << "terminate with small line search rate (<1e-8): L2-norm = " << grad.norm() << std::endl;
			break;
		}

		if (projGradNorm(x0, grad, gradTol / std::sqrt(DIM)) < gradTol)
		{
			std::cout << "terminate with gradient L2-norm = " << grad.norm() << std::endl;
			break;
		}

		if (rate * delta_x.norm() < xTol)
		{
			std::cout << "terminate with small variable change, gradient L2-norm = " << grad.norm() << std::endl;
			break;
		}

		if (f - fnew < fTol)
		{
			std::cout << "terminate with small energy change, gradient L2-norm = " << grad.norm() << std::endl;
			break;
		}
	}
	if (i >= numIter)
		std::cout << "terminate with reaching the maximum iteration, with gradient L2-norm = " << grad.norm() << std::endl;

	f = objFunc(x0, &grad, NULL, false);
	std::cout << "end up with energy: " << f << ", gradient: " << projGradNorm(x0, grad, gradTol / std::sqrt(DIM)) << std::endl;

	totalTimer.stop();
	if (displayInfo)
	{
		std::cout << "total time costed (s): " << totalTimer.elapsed<std::chrono::milliseconds>() * 1e-3 << ", within that, assembling took: " << totalAssemblingTime << ", LLT solver took: " << totalSolvingTime << ", line search took: " << totalLineSearchTime << std::endl;
	}
}

void testProjectedNewton()
{
	std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj)->double
	{
		double f = 0.5 * (x[0] - 1) * (x[0] - 1) + 0.5 * (x[1] - 2) * (x[1] - 2);
		if (grad)
		{
			grad->resize(2);
			(*grad)[0] = x[0] - 1;
			(*grad)[1] = x[1] - 2;
		}
		if (hess)
		{
			hess->resize(2, 2);
			hess->setIdentity();
		}

		return f;
	};
	std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> findMaxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) -> double
	{
		return 1.0;
	};

	Eigen::VectorXd x0(2), x(2);
	x0 << 0.3, 1.7;
	x = x0;

	std::cout << "\nProjected Newton solver: " << std::endl;
	Eigen::VectorXd lx(2), ux(2);
	lx << 1.2, 1.5;
	ux.setConstant(std::numeric_limits<double>::max());
	ProjectedNewtonSolver(objFunc, findMaxStep, lx, ux, x, 1000, 1e-6, 1e-15, 1e-15, true);
	std::cout << "only lx: " << lx.transpose() << std::endl;
	std::cout << "sol: " << x.transpose() << std::endl;

	ux << 2.3, 2.1;
	lx[0] = std::numeric_limits<double>::min();
	x = x0;
	ProjectedNewtonSolver(objFunc, findMaxStep, lx, ux, x, 1000, 1e-6, 1e-15, 1e-15, true);
	std::cout << "with ux: " << ux.transpose() << std::endl;
	std::cout << "sol: " << x.transpose() << std::endl;
}