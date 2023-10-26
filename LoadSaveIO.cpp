#include "LoadSaveIO.h"
#include <iomanip>

bool LoadEdgeOmega(const std::string& filename, const int& nlines, VectorX& edgeOmega) {
  std::ifstream infile(filename);
  if (!infile) {
    std::cerr << "invalid edge omega file name" << std::endl;
    return false;
  } else {
    Eigen::MatrixXd halfEdgeOmega(nlines, 2);
    edgeOmega.setZero(nlines);
    for (int i = 0; i < nlines; i++) {
      std::string line;
      std::getline(infile, line);
      std::stringstream ss(line);

      std::string x, y;
      ss >> x;
      if (!ss) return false;
      ss >> y;
      if (!ss) {
        halfEdgeOmega.row(i) << std::stod(x), -std::stod(x);
      } else
        halfEdgeOmega.row(i) << std::stod(x), std::stod(y);
    }
    edgeOmega = (halfEdgeOmega.col(0) - halfEdgeOmega.col(1)) / 2;
  }
  return true;
}

bool LoadVertexZvals(const std::string& filePath, const int& nlines, VectorX& zvals) {
  std::ifstream zfs(filePath);
  if (!zfs) {
    std::cerr << "invalid zvals file name" << std::endl;
    return false;
  }

  zvals.resize(2 * nlines);

  for (int j = 0; j < nlines; j++) {
    std::string line;
    std::getline(zfs, line);
    std::stringstream ss(line);
    std::string x, y;
    ss >> x;
    ss >> y;
    zvals[j] = std::stod(x);
    zvals[j + nlines] = std::stod(y);
  }
  return true;
}

bool LoadVertexAmp(const std::string& filePath, const int& nlines, VectorX& amp) {
  std::ifstream afs(filePath);

  if (!afs) {
    std::cerr << "invalid ref amp file name" << std::endl;
    return false;
  }

  amp.setZero(nlines);

  for (int j = 0; j < nlines; j++) {
    std::string line;
    std::getline(afs, line);
    std::stringstream ss(line);
    std::string x;
    ss >> x;
    if (!ss) return false;
    amp(j) = std::stod(x);
  }
  return true;
}

bool SaveEdgeOmega(const std::string& filename, const VectorX& edgeOmega) {
  std::ofstream wfs(filename);
  if (!wfs) {
    std::cerr << "invalid omega file name" << std::endl;
    return false;
  }
  wfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << edgeOmega << std::endl;
  return true;
}

bool SaveVertexZvals(const std::string& filePath, const VectorX& zvals) {
  std::ofstream zfs(filePath);
  if (!zfs) {
    std::cerr << "invalid zvals file name" << std::endl;
    return false;
  }
  assert(zvals.size() % 2 == 0);
  int nverts = zvals.size() / 2;
  for (int j = 0; j < nverts; j++) {
    zfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << zvals[j] << " " << zvals[j + nverts]
        << std::endl;
  }
  return true;
}

bool SaveVertexAmp(const std::string& filePath, const VectorX& amp) {
  std::ofstream afs(filePath);
  if (!afs) {
    std::cerr << "invalid amplitude file name" << std::endl;
    return false;
  }
  afs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << amp << std::endl;
  return true;
}

void SaveDphi4Render(const MatrixX& faceOmega, const Mesh& mesh, const std::string& filename) {
    int nfaces = mesh.GetFaceCount();
    std::ofstream dpfs(filename);

    for (int f = 0; f < nfaces; f++) {
        std::vector<int> faceVerts = mesh.GetFaceVerts(f);
        Eigen::Vector3d e0 = mesh.GetVertPos(faceVerts[1]) - mesh.GetVertPos(faceVerts[0]);
        Eigen::Vector3d e1 = mesh.GetVertPos(faceVerts[2]) - mesh.GetVertPos(faceVerts[0]);

        Eigen::Vector2d rhs;
        double u = faceOmega.row(f).dot(e0);
        double v = faceOmega.row(f).dot(e1);

        rhs << u, v;
        Eigen::Matrix2d I;
        I << e0.dot(e0), e0.dot(e1), e1.dot(e0), e1.dot(e1);
        rhs = I.inverse() * rhs;

        dpfs << rhs(0) << ",\t" << rhs(1) << ",\t" << 0 << ",\t" << 0 << ",\t" << 0 << ",\t" << 0 << std::endl;
    }
}

void SaveAmp4Render(const VectorX& vertAmp, const std::string& filename, double ampMin, double ampMax) {
    std::ofstream afs(filename);
    if (ampMin >= ampMax) ampMin = 0;
    ampMin = 0;

    for (int j = 0; j < vertAmp.rows(); j++) {
        afs << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
            << (vertAmp[j] - ampMin) / (ampMax - ampMin) << ",\t" << 3.14159 << std::endl;
    }
}


void SavePhi4Render(const VectorX& vertPhi, const std::string& fileName) {
    std::ofstream pfs(fileName);
    for (int j = 0; j < vertPhi.rows(); j++) {
        pfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << vertPhi[j] << ",\t" << 3.14159
            << std::endl;
    }
}

void SaveFlag4Render(const Eigen::VectorXi& faceFlags, const std::string& filename) {
    std::ofstream ffs(filename);

    for (int j = 0; j < faceFlags.rows(); j++) {
        ffs << faceFlags(j) << ",\t" << 3.14159 << std::endl;
    }
}

void SaveSourcePts4Render(const Eigen::VectorXi& vertFlags, const MatrixX& vertVecs, const VectorX& vertAmp,
                          const std::string& flagfilename) {
    std::ofstream ffs(flagfilename);

    for (int j = 0; j < vertFlags.rows(); j++) {
        ffs << vertFlags(j) << ",\t" << vertAmp(j) << ",\t" << vertVecs(j, 0) << ",\t" << vertVecs(j, 1) << ",\t"
            << vertVecs(j, 2) << std::endl;
    }
}
