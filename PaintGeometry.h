#pragma once

#include <Eigen/Core>

class PaintGeometry
{
public:
	PaintGeometry()
        : isNormalize(true)
	{}
	~PaintGeometry() {}
	void SetNormalization(bool flag) { isNormalize = flag; }
	Eigen::MatrixXd PaintPhi(const Eigen::VectorXd& phi, Eigen::VectorXd* brightness = NULL);    // brightness should between 0 and 1

private:
	bool isNormalize;
};

