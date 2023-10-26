#pragma once
#include <vector>

struct QuadraturePoints {
    double u;
    double v;
    double weight;
};

std::vector<QuadraturePoints> BuildQuadraturePoints(int order); // Based on the paper: http://lsec.cc.ac.cn/~tcui/myinfo/paper/quad.pdf and the corresponding source codes: http://lsec.cc.ac.cn/phg/download.htm (quad.c)