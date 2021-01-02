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

static double AdamScaling(double x) { return 1.0 / (sqrt(x) + 1.0e-8); }

class NeuralOptimizer {
public:
  NeuralOptimizer() {}
  virtual ~NeuralOptimizer() {}
  virtual void Step(std::vector<Matrix> &w, std::vector<Matrix> &dw, std::vector<Matrix> &b, std::vector<Matrix> &db) = 0;

private:

};

class Adam : public NeuralOptimizer {
public:
  Adam(const std::vector<size_t> &d, double alpha = 1.0e-3, double beta1 = 0.9, double beta2 = 0.999) : d_(d), alpha_(alpha), beta1_(beta1), beta2_(beta2), beta1_t_(beta1), beta2_t_(beta2) {
    for (size_t i = 1; i < d_.size(); i++) {
      dbm_.emplace_back(1, d_[i]);
      dbm_hat_.emplace_back(1, d_[i]);
      dbv_.emplace_back(1, d_[i]);
      dbv_hat_.emplace_back(1, d_[i]);
      dwm_.emplace_back(d_[i - 1], d_[i]);
      dwm_hat_.emplace_back(d_[i - 1], d_[i]);
      dwv_.emplace_back(d_[i - 1], d_[i]);
      dwv_hat_.emplace_back(d_[i - 1], d_[i]);
    }
  }

  virtual ~Adam() {}

  virtual void Step(std::vector<Matrix> &w, std::vector<Matrix> &dw, std::vector<Matrix> &b, std::vector<Matrix> &db) {
    for (size_t i = 0; i < d_.size() - 1; i++) {
      dbm_[i] = beta1_ * dbm_[i] + (1.0 - beta1_) * db[i];
      dwm_[i] = beta1_ * dwm_[i] + (1.0 - beta1_) * dw[i];
      dbv_[i] = beta2_ * dbv_[i] + (1.0 - beta2_) * ElementWise(db[i], db[i]);
      dwv_[i] = beta2_ * dwv_[i] + (1.0 - beta2_) * ElementWise(dw[i], dw[i]);
      dbm_hat_[i] = dbm_[i] / (1.0 - beta1_t_);
      dwm_hat_[i] = dwm_[i] / (1.0 - beta1_t_);
      dbv_hat_[i] = dbv_[i] / (1.0 - beta2_t_);
      dwv_hat_[i] = dwv_[i] / (1.0 - beta2_t_);
      beta1_t_ *= beta1_;
      beta2_t_ *= beta2_;
      b[i] -= alpha_ * ElementWise(dbm_hat_[i], dbv_hat_[i].Apply(AdamScaling));
      w[i] -= alpha_ * ElementWise(dwm_hat_[i], dwv_hat_[i].Apply(AdamScaling));
    }
  }

private:
  std::vector<size_t> d_;
  double alpha_;
  double beta1_;
  double beta2_;
  double beta1_t_;
  double beta2_t_;
  std::vector<Matrix> dbm_;
  std::vector<Matrix> dbm_hat_;
  std::vector<Matrix> dbv_;
  std::vector<Matrix> dbv_hat_;
  std::vector<Matrix> dwm_;
  std::vector<Matrix> dwm_hat_;
  std::vector<Matrix> dwv_;
  std::vector<Matrix> dwv_hat_;
};

class GradientDescent : public NeuralOptimizer {
public:
  GradientDescent(const std::vector<size_t> &d, double alpha = 1.0e-3, double mu = 0.9) : d_(d), alpha_(alpha), mu_(mu) {
    for (size_t i = 1; i < d_.size(); i++) {
      dbl_.emplace_back(1, d_[i]);
      dwl_.emplace_back(d_[i - 1], d_[i]);
    }
  }

  virtual ~GradientDescent() {}

  virtual void Step(std::vector<Matrix> &w, std::vector<Matrix> &dw, std::vector<Matrix> &b, std::vector<Matrix> &db) {
    for (size_t i = 0; i < d_.size() - 1; i++) {
      dbl_[i] = mu_ * dbl_[i] + (1.0 - mu_) * db[i];
      b[i] -= alpha_ * dbl_[i];
      dwl_[i] = mu_ * dwl_[i] + (1.0 - mu_) * dw[i];
      w[i] -= alpha_ * dwl_[i];
    }
  }
  
private:
  std::vector<size_t> d_;
  double alpha_;
  double mu_;
  std::vector<Matrix> dbl_;
  std::vector<Matrix> dwl_;
};

class NeuralNetwork {
public:
  NeuralNetwork(const std::vector<size_t> &d) : d_(d) {
    for (size_t i = 1; i < d_.size(); i++) {
      w_.emplace_back(d_[i - 1], d_[i]);
      dw_.emplace_back(d_[i - 1], d_[i]);
      b_.emplace_back(1, d_[i]);
      db_.emplace_back(1, d_[i]);
      w_.back().Random(-1.0, 1.0);
      b_.back().Random(-1.0, 1.0);
    }
  }

  void Train(Matrix &x, Matrix &y, size_t epochs, NeuralOptimizer &optimizer) {
    std::vector<Matrix> a;
    std::vector<Matrix> g;
    const int n_ops = d_.size() - 1;
    for (size_t i = 0; i <= n_ops; i++) {
      a.emplace_back(x.Rows(), d_[i]);
    }
    for (size_t i = 1; i <= n_ops; i++) {
      g.emplace_back(x.Rows(), d_[i]);
    }
    Matrix ones(x.Rows(), 1);
    ones.Ones();
    a[0] = x;
    for (size_t i = 0; i < epochs; i++) {
      for (size_t j = 1; j <= n_ops; j++) {
        a[j] = Sigmoid(a[j - 1] * w_[j - 1] + ones * b_[j - 1]);
      }
      Matrix d = a[n_ops] - y;
      g[n_ops - 1] = ElementWise(SigmoidPrime(a[n_ops]), d);
      double e = ElementWise(d, d).ReduceSum() / 2.0 / x.Rows();
      printf("%ld %e\n", i, e);
      for (int j = n_ops - 2; j >= 0; j--) {
        g[j] = ElementWise(SigmoidPrime(a[j + 1]), g[j + 1] * w_[j + 1].Transpose());
      }
      for (size_t j = 0; j < n_ops; j++) {
        dw_[j] = a[j].Transpose() * g[j] / x.Rows();
      }
      for (size_t j = 0; j < n_ops; j++) {
        db_[j] = ones.Transpose() * g[j] / x.Rows();
      }
      optimizer.Step(w_, dw_, b_, db_);
    }
  }

  const Matrix Predict(Matrix &x) {
    std::vector<Matrix> a;
    const int n_ops = d_.size() - 1;
    for (size_t i = 0; i <= n_ops; i++) {
      a.emplace_back(x.Rows(), d_[i]);
    }
    Matrix ones(x.Rows(), 1);
    ones.Ones();
    a[0] = x;
    for (size_t j = 1; j <= n_ops; j++) {
      a[j] = Sigmoid(a[j - 1] * w_[j - 1] + ones * b_[j - 1]);
    }
    return a[n_ops];
  }

private:
  std::vector<size_t> d_;
  std::vector<Matrix> b_;
  std::vector<Matrix> db_;
  std::vector<Matrix> w_;
  std::vector<Matrix> dw_;
};
