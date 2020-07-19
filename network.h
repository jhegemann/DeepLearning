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

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "dense.h"

double S(double x) { return 1.0 / (1.0 + exp(-x)); }
double SD(double s) { return s * (1.0 - s); }

class NeuralNetwork {
public:
  NeuralNetwork() {}

  virtual ~NeuralNetwork() {}

  void SetTopology(const std::vector<size_t> &d) {
    n_ = d.size() - 1;
    for (size_t i = 0; i < n_; i++) {
      d_.emplace_back(d[i + 1]);
      z_.emplace_back(d[i + 1]);
      a_.emplace_back(d[i + 1]);
      p_.emplace_back(d[i + 1]);
      b_.emplace_back(d[i + 1]);
      b_.back().Random();
      db_.emplace_back(d[i + 1]);
      dbl_.emplace_back(d[i + 1]);
      w_.emplace_back(d[i + 1], d[i]);
      w_.back().Random();
      dw_.emplace_back(d[i + 1], d[i]);
      dwl_.emplace_back(d[i + 1], d[i]);
    }
  }

  const Vector &ForwardPass(const Vector &i) {
    z_[0] = w_[0] * i + b_[0];
    a_[0] = z_[0].Apply(S);
    p_[0] = a_[0].Apply(SD);
    for (size_t i = 1; i < n_; i++) {
      z_[i] = w_[i] * a_[i - 1] + b_[i];
      a_[i] = z_[i].Apply(S);
      p_[i] = a_[i].Apply(SD);
    }
    return a_.back();
  }

  void BackwardPass(const Vector &i, const Vector &o, const Vector &t) {
    e_ = o - t;
    d_[n_ - 1] = HadamardProduct(e_, p_[n_ - 1]);
    db_[n_ - 1] = d_[n_ - 1];
    dw_[n_ - 1] = CartesianProduct(a_[n_ - 2], d_[n_ - 1]);
    for (size_t i = n_ - 2; i >= 1; i--) {
      d_[i] = HadamardProduct(w_[i + 1].Transpose() * d_[i + 1], p_[i]);
      db_[i] = d_[i];
      dw_[i] = CartesianProduct(a_[i - 1], d_[i]);
    }
    d_[0] = HadamardProduct(w_[1].Transpose() * d_[1], p_[0]);
    db_[0] = d_[0];
    dw_[0] = CartesianProduct(i, d_[0]);
  }

  void ApplyUpdate() {
    for (size_t i = 0; i < n_; i++) {
      dbl_[i] = db_[i] * (1.0 - mu_) + dbl_[i] * mu_;
      b_[i] = b_[i] - dbl_[i] * lambda_;
      dwl_[i] = dw_[i] * (1.0 - mu_) + dwl_[i] * mu_;
      w_[i] = w_[i] - dwl_[i] * lambda_;
    }
  }

  void Shuffle(std::vector<Vector> &is, std::vector<Vector> &ts) {
    assert(is.size() == ts.size());
    size_t j;
    for (size_t i = 0; i < is.size(); i++) {
      j = i + g_.Uint64() % (is.size() - i);
      std::swap(is[i], is[j]);
      std::swap(ts[i], ts[j]);
    }
  }

  void Train(std::vector<Vector> &inputs, std::vector<Vector> &targets, size_t epochs) {
    assert(inputs.size() == targets.size());
    for (size_t i = 0; i < epochs; i++) {
      Shuffle(inputs, targets);
      double error = 0.0;
      for (size_t j = 0; j < inputs.size(); j++) {
        BackwardPass(inputs[j], ForwardPass(inputs[j]), targets[j]);
        error += e_ * e_;
        ApplyUpdate();
      }
      error /= (double)inputs.size();
      printf("epoch %ld - training error: %f\n", i, error);
    }
  }

private:
  double lambda_ = 1.0e-1;
  double mu_ = 0.9;
  RandomGenerator g_;
  size_t n_;
  Vector e_;
  std::vector<Vector> d_;
  std::vector<Vector> z_;
  std::vector<Vector> a_;
  std::vector<Vector> p_;
  std::vector<Vector> b_;
  std::vector<Vector> db_;
  std::vector<Vector> dbl_;
  std::vector<Matrix> w_;
  std::vector<Matrix> dw_;
  std::vector<Matrix> dwl_;
};
