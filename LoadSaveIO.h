#pragma once
#include "CommonTools.h"
#include <iostream>
#include <fstream>


bool loadEdgeOmega(const std::string& filename, const int& nlines, VectorX& edgeOmega);
bool loadVertexZvals(const std::string& filePath, const int& nlines, ComplexVectorX& zvals);
bool loadVertexAmp(const std::string& filePath, const int& nlines, VectorX& amp);

bool saveEdgeOmega(const std::string& filename, const VectorX& edgeOmega);
bool saveVertexZvals(const std::string& filePath, const ComplexVectorX& zvals);
bool saveVertexAmp(const std::string& filePath, const VectorX& amp);