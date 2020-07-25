/* MIT License

Copyright (c) 2020 Jonas Hegemann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

class Vector;

class Matrix;

class Vector {
public:
  Vector();
  Vector(const size_t d);
  Vector(const Vector &x);
  Vector(const std::vector<double> &x);
  Vector(Vector &&x);
  Vector(std::vector<double> &&x);
  virtual ~Vector();
  void Print();
  void Shape(const size_t d);
  void Zero();
  void One();
  void Random(double min, double max);
  Vector &operator=(const Vector &x);
  Vector &operator=(const std::vector<double> &x);
  Vector &operator=(Vector &&x);
  Vector &operator=(std::vector<double> &&x);
  Vector &operator+=(const Vector &x);
  Vector &operator-=(const Vector &x);
  Vector &operator*=(const double a);
  Vector &operator/=(const double a);
  const double &operator()(const size_t i) const;
  double &operator()(const size_t i);
  Vector Apply(std::function<double(double)> f);
  double Sum() const;
  double Min() const;
  double Max() const;
  double Norm1() const;
  double Norm2() const;
  double NormInfinity() const;
  size_t Dim() const;

private:
  std::vector<double> x_;
};

Vector::Vector() { Shape(0); }

Vector::Vector(const size_t d) { Shape(d); }

Vector::Vector(const Vector &x) { x_ = x.x_; }

Vector::Vector(const std::vector<double> &x) { x_ = x; }

Vector::Vector(Vector &&x) { x_ = std::move(x.x_); }

Vector::Vector(std::vector<double> &&x) { x_ = std::move(x); }

Vector::~Vector() { x_.clear(); }

void Vector::Print() {
  printf("Vector %ld\n", x_.size());
  for (size_t i = 0; i < x_.size(); i++) {
    printf("%f\n", x_[i]);
  }
  printf("\n");
}

void Vector::Shape(const size_t d) {
  x_.resize(d);
  x_.reserve(d);
  x_.shrink_to_fit();
  Zero();
}

void Vector::Zero() {
  for (size_t i = 0; i < x_.size(); i++) {
    x_[i] = 0.0;
  }
}

void Vector::One() {
  for (size_t i = 0; i < x_.size(); i++) {
    x_[i] = 1.0;
  }
}

void Vector::Random(double min, double max) {
  for (size_t i = 0; i < x_.size(); i++) {
    x_[i] = ((double)rand() / (double)RAND_MAX) * (max - min) + min;
  }
}

Vector &Vector::operator=(const Vector &x) {
  if (this == &x) {
    return *this;
  }
  x_ = x.x_;
  return *this;
}

Vector &Vector::operator=(const std::vector<double> &x) {
  x_ = x;
  return *this;
}

Vector &Vector::operator=(Vector &&x) {
  if (this == &x) {
    return *this;
  }
  x_ = std::move(x.x_);
  return *this;
}

Vector &Vector::operator=(std::vector<double> &&x) {
  x_ = std::move(x);
  return *this;
}

Vector &Vector::operator+=(const Vector &x) {
  for (size_t i = 0; i < x_.size(); i++) {
    x_[i] += x(i);
  }
  return *this;
}

Vector &Vector::operator-=(const Vector &x) {
  for (size_t i = 0; i < x_.size(); i++) {
    x_[i] -= x(i);
  }
  return *this;
}

Vector &Vector::operator*=(const double a) {
  for (size_t i = 0; i < x_.size(); i++) {
    x_[i] *= a;
  }
  return *this;
}

Vector &Vector::operator/=(const double a) {
  for (size_t i = 0; i < x_.size(); i++) {
    x_[i] /= a;
  }
  return *this;
}

const double &Vector::operator()(const size_t i) const { return x_[i]; }

double &Vector::operator()(const size_t i) { return x_[i]; }

Vector Vector::Apply(std::function<double(double)> f) {
  Vector y(x_.size());
  for (size_t i = 0; i < x_.size(); i++) {
    y(i) = f(x_[i]);
  }
  return y;
}

double Vector::Sum() const {
  double s = x_[0];
  for (size_t i = 1; i < x_.size(); i++) {
    s += x_[i];
  }
  return s;
}

double Vector::Min() const {
  double m = x_[0];
  for (size_t i = 1; i < x_.size(); i++) {
    if (x_[i] < m) {
      m = x_[i];
    }
  }
  return m;
}

double Vector::Max() const {
  double m = x_[0];
  for (size_t i = 1; i < x_.size(); i++) {
    if (x_[i] > m) {
      m = x_[i];
    }
  }
  return m;
}

double Vector::Norm1() const {
  double n = sqrt(pow(x_[0], 2));
  for (size_t i = 1; i < x_.size(); i++) {
    n += sqrt(pow(x_[i], 2));
  }
  return n;
}

double Vector::Norm2() const {
  double n = pow(x_[0], 2);
  for (size_t i = 1; i < x_.size(); i++) {
    n += pow(x_[i], 2);
  }
  return sqrt(n);
}

double Vector::NormInfinity() const {
  double s = sqrt(pow(x_[0], 2));
  for (size_t i = 1; i < x_.size(); i++) {
    double v = sqrt(pow(x_[i], 2));
    if (v > s) {
      s = v;
    }
  }
  return s;
}

size_t Vector::Dim() const { return x_.size(); }

class Matrix {
public:
  Matrix();
  Matrix(const size_t m, const size_t n);
  Matrix(const Matrix &x);
  Matrix(Matrix &&x);
  virtual ~Matrix();
  void Print();
  void Shape(const size_t m, const size_t n);
  void Zero();
  void One();
  void Random(double min, double max);
  Matrix Apply(std::function<double(double)> f);
  Matrix &operator=(const Matrix &x);
  Matrix &operator=(Matrix &&x);
  const double &operator()(const size_t i, const size_t j) const;
  double &operator()(const size_t i, const size_t j);
  Matrix &operator+=(const Matrix &x);
  Matrix &operator-=(const Matrix &x);
  Matrix &operator*=(const double a);
  Matrix &operator/=(const double a);
  double Trace() const;
  size_t Rows() const;
  size_t Cols() const;
  const Matrix Transpose() const;

private:
  std::vector<double> x_;
  size_t m_;
  size_t n_;
};

Matrix::Matrix() { Shape(0, 0); }

Matrix::Matrix(const size_t m, const size_t n) { Shape(m, n); }

Matrix::Matrix(const Matrix &x) {
  x_ = x.x_;
  m_ = x.m_;
  n_ = x.n_;
}

Matrix::Matrix(Matrix &&x) { x_ = std::move(x.x_); }

Matrix::~Matrix() { x_.clear(); }

void Matrix::Print() {
  printf("Matrix (%ld x %ld)\n", m_, n_);
  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      printf("%f\t", (*this)(i, j));
    }
    printf("\n");
  }
  printf("\n");
}

void Matrix::Shape(const size_t m, const size_t n) {
  m_ = m;
  n_ = n;
  x_.resize(m_ * n_);
  x_.reserve(m_ * n_);
  x_.shrink_to_fit();
  Zero();
}

void Matrix::Zero() {
  for (size_t i = 0; i < x_.size(); i++) {
    x_[i] = 0.0;
  }
}

void Matrix::One() {
  Zero();
  for (size_t i = 0; i < m_; i++) {
    (*this)(i, i) = 1.0;
  }
}

void Matrix::Random(double min, double max) {
  for (size_t i = 0; i < x_.size(); i++) {
    x_[i] = ((double)rand() / (double)RAND_MAX) * (max - min) + min;
  }
}

Matrix Matrix::Apply(std::function<double(double)> f) {
  Matrix x(m_, n_);
  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      x(i, j) = f((*this)(i, j));
    }
  }
  return x;
}

Matrix &Matrix::operator=(const Matrix &x) {
  if (this == &x) {
    return *this;
  }
  x_ = x.x_;
  m_ = x.m_;
  n_ = x.n_;
  return *this;
}

Matrix &Matrix::operator=(Matrix &&x) {
  if (this == &x) {
    return *this;
  }
  x_ = std::move(x.x_);
  m_ = std::move(x.m_);
  n_ = std::move(x.n_);
  return *this;
}

const double &Matrix::operator()(const size_t i, const size_t j) const {
  return x_[i * n_ + j];
}

double &Matrix::operator()(const size_t i, const size_t j) {
  return x_[i * n_ + j];
}

Matrix &Matrix::operator+=(const Matrix &x) {
  for (size_t i = 0; i < x.Rows(); i++) {
    for (size_t j = 0; j < x.Cols(); j++) {
      (*this)(i, j) += x(i, j);
    }
  }
  return *this;
}

Matrix &Matrix::operator-=(const Matrix &x) {
  for (size_t i = 0; i < x.Rows(); i++) {
    for (size_t j = 0; j < x.Cols(); j++) {
      (*this)(i, j) -= x(i, j);
    }
  }
  return *this;
}

Matrix &Matrix::operator*=(const double a) {
  for (size_t i = 0; i < Rows(); i++) {
    for (size_t j = 0; j < Cols(); j++) {
      (*this)(i, j) *= a;
    }
  }
  return *this;
}

Matrix &Matrix::operator/=(const double a) {
  for (size_t i = 0; i < Rows(); i++) {
    for (size_t j = 0; j < Cols(); j++) {
      (*this)(i, j) /= a;
    }
  }
  return *this;
}

double Matrix::Trace() const {
  double t = (*this)(0, 0);
  for (size_t i = 1; i < m_; i++) {
    t += (*this)(i, i);
  }
  return t;
}

size_t Matrix::Rows() const { return m_; }

size_t Matrix::Cols() const { return n_; }

const Matrix Matrix::Transpose() const {
  Matrix t(n_, m_);
  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      t(j, i) = (*this)(i, j);
    }
  }
  return t;
}

const Vector operator/(const Vector &x, const double a) {
  Vector s(x);
  for (size_t i = 0; i < s.Dim(); i++) {
    s(i) /= a;
  }
  return s;
}

const Vector operator*(const Vector &x, const double a) {
  Vector s(x);
  for (size_t i = 0; i < s.Dim(); i++) {
    s(i) *= a;
  }
  return s;
}

const Vector operator*(const double a, const Vector &x) { return x * a; }

const double operator*(const Vector &x, const Vector &y) {
  double p = 0.0;
  for (size_t i = 0; i < x.Dim(); i++) {
    p += x(i) * y(i);
  }
  return p;
}

const Vector operator+(const Vector &x, const Vector &y) {
  Vector s(x);
  for (size_t i = 0; i < x.Dim(); i++) {
    s(i) += y(i);
  }
  return s;
}

const Vector operator-(const Vector &x, const Vector &y) {
  Vector s(x);
  for (size_t i = 0; i < x.Dim(); i++) {
    s(i) -= y(i);
  }
  return s;
}

const Matrix operator/(const Matrix &m, const double a) {
  Matrix s(m);
  for (size_t i = 0; i < s.Rows(); i++) {
    for (size_t j = 0; j < s.Cols(); j++) {
      s(i, j) /= a;
    }
  }
  return s;
}

const Matrix operator*(const Matrix &m, const double a) {
  Matrix s(m);
  for (size_t i = 0; i < s.Rows(); i++) {
    for (size_t j = 0; j < s.Cols(); j++) {
      s(i, j) *= a;
    }
  }
  return s;
}

const Matrix operator*(const double a, const Matrix &m) { return m * a; }

const Matrix operator*(const Matrix &m, const Matrix &n) {
  Matrix p(m.Rows(), n.Cols());
  for (size_t i = 0; i < m.Rows(); i++) {
    for (size_t j = 0; j < n.Cols(); j++) {
      for (size_t k = 0; k < m.Cols(); k++) {
        p(i, j) += m(i, k) * n(k, j);
      }
    }
  }
  return p;
}

const Matrix operator+(const Matrix &m, const Matrix &n) {
  Matrix a(m);
  for (size_t i = 0; i < a.Rows(); i++) {
    for (size_t j = 0; j < a.Cols(); j++) {
      a(i, j) += n(i, j);
    }
  }
  return a;
}

const Matrix operator-(const Matrix &m, const Matrix &n) {
  Matrix s(m);
  for (size_t i = 0; i < s.Rows(); i++) {
    for (size_t j = 0; j < s.Cols(); j++) {
      s(i, j) -= n(i, j);
    }
  }
  return s;
}

const Vector operator*(const Matrix &m, const Vector &x) {
  Vector p(m.Rows());
  for (size_t i = 0; i < m.Rows(); i++) {
    for (size_t j = 0; j < m.Cols(); j++) {
      p(i) += m(i, j) * x(j);
    }
  }
  return p;
}

const Vector HadamardProduct(const Vector &x, const Vector &y) {
  Vector p(x.Dim());
  for (size_t i = 0; i < p.Dim(); i++) {
    p(i) = x(i) * y(i);
  }
  return p;
}

const Matrix HadamardProduct(const Matrix &x, const Matrix &y) {
  Matrix p(x.Rows(), x.Cols());
  for (size_t i = 0; i < p.Rows(); i++) {
    for (size_t j = 0; j < p.Cols(); j++) {
      p(i, j) = x(i, j) * y(i, j);
    }
  }
  return p;
}

const Matrix OuterProduct(const Vector &x, const Vector &y) {
  Matrix p(x.Dim(), y.Dim());
  for (size_t i = 0; i < x.Dim(); i++) {
    for (size_t j = 0; j < y.Dim(); j++) {
      p(i, j) = x(i) * y(j);
    }
  }
  return p;
}

