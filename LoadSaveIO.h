#pragma once
#include "CommonTools.h"
#include <iostream>
#include <fstream>


bool LoadEdgeOmega(const std::string& filename, const int& nlines, VectorX& edgeOmega);
bool LoadVertexZvals(const std::string& filePath, const int& nlines, VectorX& zvals);
bool LoadVertexAmp(const std::string& filePath, const int& nlines, VectorX& amp);

bool SaveEdgeOmega(const std::string& filename, const VectorX& edgeOmega);
bool SaveVertexZvals(const std::string& filePath, const VectorX& zvals);
bool SaveVertexAmp(const std::string& filePath, const VectorX& amp);

// Save for render
void SaveDphi4Render(const MatrixX& faceOmega, const Mesh& secMesh, const std::string& filename);
void SaveAmp4Render(const VectorX& vertAmp, const std::string& filename, double ampMin = 0, double ampMax = 1);
void SavePhi4Render(const VectorX& vertPhi, const std::string& fileName);
void SaveFlag4Render(const Eigen::VectorXi& faceFlags, const std::string& filename); // 1 for selected, -1 for interface, 0 for unchanged
void SaveSourcePts4Render(const Eigen::VectorXi& vertFlags, const MatrixX& vertVecs, const VectorX& vertAmp, const std::string& flagfilename);