#include "LoadSaveIO.h"
#include <iomanip>

bool loadEdgeOmega(const std::string& filename, const int& nlines, VectorX& edgeOmega)
{
	std::ifstream infile(filename);
	if (!infile)
	{
		std::cerr << "invalid edge omega file name" << std::endl;
		return false;
	}
	else
	{
		Eigen::MatrixXd halfEdgeOmega(nlines, 2);
		edgeOmega.setZero(nlines);
		for (int i = 0; i < nlines; i++)
		{
			std::string line;
			std::getline(infile, line);
			std::stringstream ss(line);

			std::string x, y;
			ss >> x;
			if (!ss)
				return false;
			ss >> y;
			if (!ss)
			{
				halfEdgeOmega.row(i) << std::stod(x), -std::stod(x);
			}
			else
				halfEdgeOmega.row(i) << std::stod(x), std::stod(y);
		}
		edgeOmega = (halfEdgeOmega.col(0) - halfEdgeOmega.col(1)) / 2;
	}
	return true;
}

bool loadVertexZvals(const std::string& filePath, const int& nlines, ComplexVectorX& zvals)
{
	std::ifstream zfs(filePath);
	if (!zfs)
	{
		std::cerr << "invalid zvals file name" << std::endl;
		return false;
	}

	zvals.resize(nlines);

	for (int j = 0; j < nlines; j++) {
		std::string line;
		std::getline(zfs, line);
		std::stringstream ss(line);
		std::string x, y;
		ss >> x;
		ss >> y;
		zvals[j] = std::complex<Scalar>(std::stod(x), std::stod(y));
	}
	return true;
}

bool loadVertexAmp(const std::string& filePath, const int& nlines, VectorX& amp)
{
	std::ifstream afs(filePath);

	if (!afs)
	{
		std::cerr << "invalid ref amp file name" << std::endl;
		return false;
	}

	amp.setZero(nlines);

	for (int j = 0; j < nlines; j++)
	{
		std::string line;
		std::getline(afs, line);
		std::stringstream ss(line);
		std::string x;
		ss >> x;
		if (!ss)
			return false;
		amp(j) = std::stod(x);
	}
	return true;
}

bool saveEdgeOmega(const std::string& filename, const VectorX& edgeOmega)
{
    std::ofstream wfs(filename);
    if(!wfs)
    {
        std::cerr << "invalid omega file name" << std::endl;
        return false;
    }
    wfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << edgeOmega << std::endl;
    return true;
}

bool saveVertexZvals(const std::string& filePath, const ComplexVectorX& zvals)
{
    std::ofstream zfs(filePath);
    if(!zfs)
    {
        std::cerr << "invalid zvals file name" << std::endl;
        return false;
    }
    for (int j = 0; j < zvals.size(); j++)
    {
        zfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << zvals[j].real() << " " << zvals[j].imag() << std::endl;
    }
    return true;

}
bool saveVertexAmp(const std::string& filePath, const VectorX& amp)
{
    std::ofstream  afs(filePath);
    if(!afs)
    {
        std::cerr << "invalid amplitude file name" << std::endl;
        return false;
    }
    afs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << amp << std::endl;
    return true;
}