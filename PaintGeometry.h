#pragma once

#include <Eigen/Core>

class PaintGeometry
{
public:
	PaintGeometry()
        : isNormalize(true)
	{}
	~PaintGeometry() {}
	void setNormalization(bool flag) { isNormalize = flag; }
	Eigen::MatrixXd paintPhi(const Eigen::VectorXd& phi, Eigen::VectorXd* brightness = NULL);    // brightness should between 0 and 1

private:
	bool isNormalize;
};

