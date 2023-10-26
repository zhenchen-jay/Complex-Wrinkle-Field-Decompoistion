#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iterator>

typedef double Scalar;
typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;
typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;

typedef Eigen::Triplet<Scalar> TripletX;
typedef Eigen::SparseMatrix<Scalar> SparseMatrixX;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
typedef std::back_insert_iterator< std::vector<TripletX> > TripletInserter;
