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

#define OPTIMIZER_SGD 0
#define OPTIMIZER_ADAM 1

static double S(double x) { return 1.0 / (1.0 + exp(-x)); }
static double SD(double s) { return s * (1.0 - s); }

class NeuralNetwork {
public:
  NeuralNetwork() {}

  virtual ~NeuralNetwork() {}

  void Init(const std::vector<size_t> &d) {
    beta1_t_ = beta1_;
    beta2_t_ = beta2_;
    g_.Seed(time(nullptr));
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
      dbl_.emplace_back(d[i + 1]);
      w_.emplace_back(d[i + 1], d[i]);
      w_.back().Random(g_, -0.1, 0.1);
      dw_.emplace_back(d[i + 1], d[i]);
      dwm_.emplace_back(d[i + 1], d[i]);
      dwm_hat_.emplace_back(d[i + 1], d[i]);
      dwv_.emplace_back(d[i + 1], d[i]);
      dwv_hat_.emplace_back(d[i + 1], d[i]);
      dwl_.emplace_back(d[i + 1], d[i]);
    }
  }

  void ForwardPass(const Vector &input) {
    i_ = input;
    z_[0] = w_[0] * i_ + b_[0];
    a_[0] = z_[0].Apply(S);
    p_[0] = a_[0].Apply(SD);
    for (size_t i = 1; i < n_; i++) {
      z_[i] = w_[i] * a_[i - 1] + b_[i];
      a_[i] = z_[i].Apply(S);
      p_[i] = a_[i].Apply(SD);
    }
  }

  const Vector &Predict(const Vector &input) {
    ForwardPass(input);
    return a_.back();
  }

  void BackwardPass(const Vector &target) {
    e_ = a_.back() - target;
    rms_ += e_ * e_;
    d_[n_ - 1] = HadamardProduct(e_, p_[n_ - 1]);
    db_[n_ - 1] = db_[n_ - 1] + d_[n_ - 1];
    dw_[n_ - 1] = dw_[n_ - 1] + OuterProduct(d_[n_ - 1], a_[n_ - 2]);
    for (size_t i = n_ - 2; i >= 1; i--) {
      d_[i] = HadamardProduct(w_[i + 1].Transpose() * d_[i + 1], p_[i]);
      db_[i] = db_[i] + d_[i];
      dw_[i] = dw_[i] + OuterProduct(d_[i], a_[i - 1]);
    }
    d_[0] = HadamardProduct(w_[1].Transpose() * d_[1], p_[0]);
    db_[0] = db_[0] + d_[0];
    dw_[0] = dw_[0] + OuterProduct(d_[0], i_);
  }

  void PrepareStep() {
    rms_ = 0.0;
    for (size_t i = 0; i < n_; i++) {
      db_[i].Zero();
      dw_[i].Zero();
    }
  }

  void SGD(size_t batch_size) {
    for (size_t i = 0; i < n_; i++) {
      dbl_[i] = mu_ * dbl_[i] + (1.0 - mu_) * db_[i] * (1.0 / batch_size) ;
      b_[i] = b_[i] - alpha_ * dbl_[i];
      dwl_[i] = mu_ * dwl_[i] + (1.0 - mu_) * dw_[i] * (1.0 / batch_size);
      w_[i] = w_[i] - alpha_ * dwl_[i];
    }
  }

  void Adam(size_t batch_size) {
    for (size_t i = 0; i < n_; i++) {
      dbm_[i] = beta1_ * dbm_[i] + (1.0 - beta1_) * db_[i] * (1.0 / batch_size);
      dwm_[i] = beta1_ * dwm_[i] + (1.0 - beta1_) * dw_[i] * (1.0 / batch_size);
      dbv_[i] = beta2_ * dbv_[i] + (1.0 - beta2_) * (1.0 / pow(batch_size, 2)) * HadamardProduct(db_[i], db_[i]);
      dwv_[i] = beta2_ * dwv_[i] + (1.0 - beta2_) * (1.0 / pow(batch_size, 2)) * HadamardProduct(dw_[i], dw_[i]);
      dbm_hat_[i] = dbm_[i] * (1.0 / (1.0 - beta1_t_));
      dwm_hat_[i] = dwm_[i] * (1.0 / (1.0 - beta1_t_));
      dbv_hat_[i] = dbv_[i] * (1.0 / (1.0 - beta2_t_));
      dwv_hat_[i] = dwv_[i] * (1.0 / (1.0 - beta2_t_));
      beta1_t_ *= beta1_;
      beta2_t_ *= beta2_;
      b_[i] = b_[i] - alpha_ * HadamardProduct(dbm_hat_[i], dbv_hat_[i].Apply([](double x) -> double { return 1.0 / (sqrt(x) + 1.0e-8); }));
      w_[i] = w_[i] - alpha_ * HadamardProduct(dwm_hat_[i], dwv_hat_[i].Apply([](double x) -> double { return 1.0 / (sqrt(x) + 1.0e-8); }));
    }
  }

  double Error() {
    return rms_;
  }

  void PrepareBatches(std::vector<Vector> &inputs) {
    indices_.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      indices_[i] = i;
    }
  }

  void SampleBatches(size_t batch_size) {
    for (size_t i = 0; i < indices_.size(); i++) {
      size_t j = g_.Uint64() % (indices_.size() - i) + i;
      std::swap(indices_[i], indices_[j]);
    }
    size_t count = indices_.size() / batch_size;
    size_t residual = indices_.size() % batch_size;
    batches_.resize(count);
    for (size_t i = 0; i < count; i++) {
      std::vector<size_t> batch(batch_size);
      for (size_t j = 0; j < batch_size; j++) {
        batch[j] = indices_[i * batch_size + j];
      }
      batches_[i] = batch;
    } 
    if (residual > 0) {
      std::vector<size_t> batch(residual); 
      for (size_t i = 0; i < residual; i++) {
        batch[i] = indices_[count * batch_size + i];
      }
      batches_.emplace_back(batch);
    }
  }

  void Train(std::vector<Vector> &inputs, std::vector<Vector> &targets, size_t epochs, size_t batch_size, int optimizer) {
    Vector output;
    PrepareBatches(inputs);
    for (size_t i = 0; i < epochs; i++) {
      SampleBatches(batch_size);
      double error = 0.0;
      for (size_t i = 0; i < batches_.size(); i++) {
        PrepareStep();
        std::vector<size_t> batch = batches_[i];
        for (size_t j = 0; j < batch.size(); j++) {
          ForwardPass(inputs[batch[j]]);
          BackwardPass(targets[batch[j]]);
        }
        if (optimizer == OPTIMIZER_SGD) {
          SGD(batch_size);
        } else if (optimizer == OPTIMIZER_ADAM) {
          Adam(batch_size);
        } else {
          abort();
        }
        error += Error();
      }
      double total_error = error / inputs.size();
      printf("epoch %ld - training error: %e\n", i, total_error);
      
    }
  }

private:
  RandomGenerator g_;
  const double alpha_ = 1.0e-3;
  const double beta1_ = 0.9;
  const double beta2_ = 0.999;
  double beta1_t_;
  double beta2_t_;
  const double mu_ = 0.9;
  double rms_;
  size_t n_;
  Vector i_;
  Vector t_;
  Vector e_;
  std::vector<size_t> indices_;
  std::vector<std::vector<size_t>> batches_;
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
  std::vector<Vector> dbl_;
  std::vector<Matrix> w_;
  std::vector<Matrix> dw_;
  std::vector<Matrix> dwm_;
  std::vector<Matrix> dwm_hat_;
  std::vector<Matrix> dwv_;
  std::vector<Matrix> dwv_hat_;
  std::vector<Matrix> dwl_;
};
