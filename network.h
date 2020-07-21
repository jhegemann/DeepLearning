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
#include <ctime>
#include "dense.h"

static double S(double x) { return 1.0 / (1.0 + exp(-x)); }
static double SD(double s) { return s * (1.0 - s); }

class NeuralNetwork {
public:
  NeuralNetwork() {
    g_.Seed(time(nullptr));
  }

  virtual ~NeuralNetwork() {}

  void SetTopology(const std::vector<size_t> &d) {
    n_ = d.size() - 1;
    for (size_t i = 0; i < n_; i++) {
      d_.emplace_back(d[i + 1]);
      z_.emplace_back(d[i + 1]);
      a_.emplace_back(d[i + 1]);
      p_.emplace_back(d[i + 1]);
      b_.emplace_back(d[i + 1]);
      b_.back().Random(g_, -0.1, 0.1);
      db_.emplace_back(d[i + 1]);
      dbm_.emplace_back(d[i + 1]);
      dbm_hat_.emplace_back(d[i + 1]);
      dbv_.emplace_back(d[i + 1]);
      dbv_hat_.emplace_back(d[i + 1]);
      w_.emplace_back(d[i + 1], d[i]);
      w_.back().Random(g_, -0.1, 0.1);
      dw_.emplace_back(d[i + 1], d[i]);
      dwm_.emplace_back(d[i + 1], d[i]);
      dwm_hat_.emplace_back(d[i + 1], d[i]);
      dwv_.emplace_back(d[i + 1], d[i]);
      dwv_hat_.emplace_back(d[i + 1], d[i]);
    }
  }

  const Vector &Output() {
    return a_.back();
  }

  void ForwardPass(const Vector &in) {
    z_[0] = w_[0] * in + b_[0];
    a_[0] = z_[0].Apply(S);
    p_[0] = a_[0].Apply(SD);
    for (size_t i = 1; i < n_; i++) {
      z_[i] = w_[i] * a_[i - 1] + b_[i];
      a_[i] = z_[i].Apply(S);
      p_[i] = a_[i].Apply(SD);
    }
  }

  void BackwardPass(const Vector &in, const Vector &o, const Vector &t, size_t n) {
    e_ = o - t;
    d_[n_ - 1] = HadamardProduct(e_, p_[n_ - 1]);
    db_[n_ - 1] = db_[n_ - 1] + (1.0 / n) * d_[n_ - 1];
    dw_[n_ - 1] = dw_[n_ - 1] + (1.0 / n) * DyadicProduct(d_[n_ - 1], a_[n_ - 2]);
    for (size_t i = n_ - 2; i >= 1; i--) {
      d_[i] = HadamardProduct(w_[i + 1].Transpose() * d_[i + 1], p_[i]);
      db_[i] = db_[i] + (1.0 / n) * d_[i];
      dw_[i] = dw_[i] + (1.0 / n) * DyadicProduct(d_[i], a_[i - 1]);
    }
    d_[0] = HadamardProduct(w_[1].Transpose() * d_[1], p_[0]);
    db_[0] = db_[0] + (1.0 / n) * d_[0];
    dw_[0] = dw_[0] + (1.0 / n) * DyadicProduct(d_[0], in);
  }

  void ZeroUpdate() {
    for (size_t i = 0; i < n_; i++) {
      db_[i].Zero();
      dw_[i].Zero();
    }
  }

  void ApplyUpdate() {
    for (size_t i = 0; i < n_; i++) {
      dbm_[i] = beta1_ * dbm_[i] + (1.0 - beta1_) * db_[i];
      dwm_[i] = beta1_ * dwm_[i] + (1.0 - beta1_) * dw_[i];
      dbv_[i] = beta2_ * dbv_[i] + (1.0 - beta2_) * HadamardProduct(db_[i], db_[i]);
      dwv_[i] = beta2_ * dwv_[i] + (1.0 - beta2_) * HadamardProduct(dw_[i], dw_[i]);
      dbm_hat_[i] = dbm_[i] * (1.0 / (1.0 - beta1_t_));
      dwm_hat_[i] = dwm_[i] * (1.0 / (1.0 - beta1_t_));
      dbv_hat_[i] = dbv_[i] * (1.0 / (1.0 - beta2_t_));
      dwv_hat_[i] = dwv_[i] * (1.0 / (1.0 - beta2_t_));
      beta1_t_ *= beta1_;
      beta2_t_ *= beta2_;
    }
    for (size_t i = 0; i < n_; i++) {
      b_[i] = b_[i] - alpha_ * HadamardProduct(dbm_hat_[i], dbv_hat_[i].Apply([](double x) -> double { return 1.0 / (sqrt(x) + 1.0e-8); }));
      w_[i] = w_[i] - alpha_ * HadamardProduct(dwm_hat_[i], dwv_hat_[i].Apply([](double x) -> double { return 1.0 / (sqrt(x) + 1.0e-8); }));
    }
  }

  void Train(std::vector<Vector> &inputs, std::vector<Vector> &targets, size_t epochs) {
    beta1_t_ = beta1_;
    beta2_t_ = beta2_;
    Vector output;
    for (size_t i = 0; i < epochs; i++) {
      ZeroUpdate();
      double error = 0.0;
      for (size_t j = 0; j < inputs.size(); j++) {
        ForwardPass(inputs[j]);
        BackwardPass(inputs[j], Output(), targets[j], inputs.size());
        error += e_ * e_;
      }
      error /= (double)inputs.size();
      printf("epoch %ld - training error: %e\n", i, error);
      ApplyUpdate();
    }
  }

private:
  RandomGenerator g_;
  const double alpha_ = 1.0e-3;
  const double beta1_ = 0.9;
  const double beta2_ = 0.999;
  double beta1_t_;
  double beta2_t_;
  size_t n_;
  Vector e_;
  std::vector<Vector> d_;
  std::vector<Vector> z_;
  std::vector<Vector> a_;
  std::vector<Vector> p_;
  std::vector<Vector> b_;
  std::vector<Vector> db_;
  std::vector<Vector> dbm_;
  std::vector<Vector> dbm_hat_;
  std::vector<Vector> dbv_;
  std::vector<Vector> dbv_hat_;
  std::vector<Matrix> w_;
  std::vector<Matrix> dw_;
  std::vector<Matrix> dwm_;
  std::vector<Matrix> dwm_hat_;
  std::vector<Matrix> dwv_;
  std::vector<Matrix> dwv_hat_;
};
