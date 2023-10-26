#include <igl/hsv_to_rgb.h>
#include <igl/jet.h>
#include <memory>


#include "PaintGeometry.h"


Eigen::MatrixXd PaintGeometry::paintPhi(const Eigen::VectorXd& phi,
                                        Eigen::VectorXd* brightness) // brightness should between 0 and 1
{
  int nverts = phi.size();
  // std::cout << phi.minCoeff() << " " << phi.maxCoeff() << std::endl;
  Eigen::MatrixXd color(nverts, 3);
  if (isNormalize) {
    igl::jet(phi, true, color);
  } else {
    for (int i = 0; i < nverts; i++) {
      double r, g, b;
      //            double h = 360.0 * phi[i] / 2.0 / M_PI + 120;
      double h = 360.0 * phi[i] / 2.0 / M_PI;
      h = 360 + ((int)h % 360); // fix for libigl bug
      double s = 1.0;
      double v = 0.5;
      if (brightness) {
        double r = (*brightness)(i);
        v = r * r / (r * r + 1);
      }
      //                v = (*brightness)(i);
      igl::hsv_to_rgb(h, s, v, r, g, b);
      color(i, 0) = r;
      color(i, 1) = g;
      color(i, 2) = b;
    }
  }

  return color;
}
