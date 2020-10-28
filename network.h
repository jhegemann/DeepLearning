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

static double Sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double SigmoidPrime(double s) { return s * (1.0 - s); }
static double AdamScaling(double x) { return 1.0 / (sqrt(x) + 1.0e-8); }

class NeuralOptimizer {
public:
  NeuralOptimizer() {}
  virtual ~NeuralOptimizer() {}
  virtual void Step(std::vector<Matrix> &w, std::vector<Matrix> &dw, std::vector<Vector> &b, std::vector<Vector> &db) = 0;

private:

};

class Adam : public NeuralOptimizer {
public:
  Adam(const std::vector<size_t> &d, double alpha = 1.0e-3, double beta1 = 0.9, double beta2 = 0.999) : n_(d.size() - 1), alpha_(alpha), beta1_(beta1), beta2_(beta2), beta1_t_(beta1), beta2_t_(beta2) {
    for (size_t i = 0; i < n_; i++) {
      dbm_.emplace_back(d[i + 1]);
      dbm_hat_.emplace_back(d[i + 1]);
      dbv_.emplace_back(d[i + 1]);
      dbv_hat_.emplace_back(d[i + 1]);
      dwm_.emplace_back(d[i + 1], d[i]);
      dwm_hat_.emplace_back(d[i + 1], d[i]);
      dwv_.emplace_back(d[i + 1], d[i]);
      dwv_hat_.emplace_back(d[i + 1], d[i]);
    }
  }

  virtual ~Adam() {}

  virtual void Step(std::vector<Matrix> &w, std::vector<Matrix> &dw, std::vector<Vector> &b, std::vector<Vector> &db) {
    for (size_t i = 0; i < n_; i++) {
      dbm_[i] = beta1_ * dbm_[i] + (1.0 - beta1_) * db[i];
      dwm_[i] = beta1_ * dwm_[i] + (1.0 - beta1_) * dw[i];
      dbv_[i] = beta2_ * dbv_[i] + (1.0 - beta2_) * HadamardProduct(db[i], db[i]);
      dwv_[i] = beta2_ * dwv_[i] + (1.0 - beta2_) * HadamardProduct(dw[i], dw[i]);
      dbm_hat_[i] = dbm_[i] / (1.0 - beta1_t_);
      dwm_hat_[i] = dwm_[i] / (1.0 - beta1_t_);
      dbv_hat_[i] = dbv_[i] / (1.0 - beta2_t_);
      dwv_hat_[i] = dwv_[i] / (1.0 - beta2_t_);
      beta1_t_ *= beta1_;
      beta2_t_ *= beta2_;
      b[i] -= alpha_ * HadamardProduct(dbm_hat_[i], dbv_hat_[i].Apply(AdamScaling));
      w[i] -= alpha_ * HadamardProduct(dwm_hat_[i], dwv_hat_[i].Apply(AdamScaling));
    }
  }

private:
  size_t n_;
  double alpha_;
  double beta1_;
  double beta2_;
  double beta1_t_;
  double beta2_t_;
  std::vector<Vector> dbm_;
  std::vector<Vector> dbm_hat_;
  std::vector<Vector> dbv_;
  std::vector<Vector> dbv_hat_;
  std::vector<Matrix> dwm_;
  std::vector<Matrix> dwm_hat_;
  std::vector<Matrix> dwv_;
  std::vector<Matrix> dwv_hat_;
};

class GradientDescent : public NeuralOptimizer {
public:
  GradientDescent(const std::vector<size_t> &d, double alpha = 1.0e-3, double mu = 0.9) : n_(d.size() - 1), alpha_(alpha), mu_(mu) {
    for (size_t i = 0; i < n_; i++) {
      dbl_.emplace_back(d[i + 1]);
      dwl_.emplace_back(d[i + 1], d[i]);
    }
  }

  virtual ~GradientDescent() {}

  virtual void Step(std::vector<Matrix> &w, std::vector<Matrix> &dw, std::vector<Vector> &b, std::vector<Vector> &db) {
    for (size_t i = 0; i < n_; i++) {
      dbl_[i] = mu_ * dbl_[i] + (1.0 - mu_) * db[i];
      b[i] -= alpha_ * dbl_[i];
      dwl_[i] = mu_ * dwl_[i] + (1.0 - mu_) * dw[i];
      w[i] -= alpha_ * dwl_[i];
    }
  }
  
private:
  size_t n_;
  double alpha_;
  double mu_;
  std::vector<Vector> dbl_;
  std::vector<Matrix> dwl_;
};

class NeuralNetwork {
public:
  NeuralNetwork(const std::vector<size_t> &d) : n_(d.size() - 1) {
    srand(time(nullptr));
    for (size_t i = 0; i < n_; i++) {
      d_.emplace_back(d[i + 1]);
      z_.emplace_back(d[i + 1]);
      a_.emplace_back(d[i + 1]);
      p_.emplace_back(d[i + 1]);
      b_.emplace_back(d[i + 1]);
      b_.back().Random(-0.1, 0.1);
      db_.emplace_back(d[i + 1]);
      w_.emplace_back(d[i + 1], d[i]);
      w_.back().Random(-0.1, 0.1);
      dw_.emplace_back(d[i + 1], d[i]);
    }
  }

  virtual ~NeuralNetwork() {}

  void ForwardPass(const Vector &input) {
    i_ = input;
    z_[0] = w_[0] * i_ + b_[0];
    a_[0] = z_[0].Apply(Sigmoid);
    p_[0] = a_[0].Apply(SigmoidPrime);
    for (size_t i = 1; i < n_; i++) {
      z_[i] = w_[i] * a_[i - 1] + b_[i];
      a_[i] = z_[i].Apply(Sigmoid);
      p_[i] = a_[i].Apply(SigmoidPrime);
    }
  }

  const Vector &Prediction() {
    return a_.back();
  }

  void BackwardPass(const Vector &target) {
    e_ = a_.back() - target;
    rms_ += e_ * e_;
    d_[n_ - 1] = HadamardProduct(e_, p_[n_ - 1]);
    db_[n_ - 1] += d_[n_ - 1];
    dw_[n_ - 1] += OuterProduct(d_[n_ - 1], a_[n_ - 2]);
    for (size_t i = n_ - 2; i >= 1; i--) {
      d_[i] = HadamardProduct(w_[i + 1].Transpose() * d_[i + 1], p_[i]);
      db_[i] += d_[i];
      dw_[i] += OuterProduct(d_[i], a_[i - 1]);
    }
    d_[0] = HadamardProduct(w_[1].Transpose() * d_[1], p_[0]);
    db_[0] += d_[0];
    dw_[0] += OuterProduct(d_[0], i_);
  }

  void ScaleGradient(size_t batch_size) {
    for (size_t i = 0; i < n_; i++) {
      db_[i] /= batch_size;
      dw_[i] /= batch_size;
    }
  }

  void ZeroGradient() {
    rms_ = 0.0;
    for (size_t i = 0; i < n_; i++) {
      db_[i].Zero();
      dw_[i].Zero();
    }
  }

  double Error() {
    return rms_;
  }

  void PrepareBatches(std::vector<Vector> &inputs, std::vector<Vector> &targets, size_t batch_size) {
    major_batch_size_ = batch_size;
    dataset_indices_.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      dataset_indices_[i] = i;
    }
    batches_count_ = dataset_indices_.size() / major_batch_size_;
    batches_residual_ = dataset_indices_.size() % major_batch_size_;
    for (size_t i = 0; i < batches_count_; i++) {
      batches_.emplace_back(major_batch_size_);
    } 
    if (batches_residual_ > 0) {
      batches_.emplace_back(batches_residual_);
    }
  }

  void SampleBatches() {
    for (size_t i = 0; i < dataset_indices_.size(); i++) {
      size_t j = rand() % (dataset_indices_.size() - i) + i;
      std::swap(dataset_indices_[i], dataset_indices_[j]);
    }
    for (size_t i = 0; i < batches_count_; i++) {
      for (size_t j = 0; j < major_batch_size_; j++) {
        batches_[i][j] = dataset_indices_[i * major_batch_size_ + j];
      }
    } 
    for (size_t i = 0; i < batches_residual_; i++) {
      batches_.back()[i] = dataset_indices_[batches_count_ * major_batch_size_ + i];
    }
  }

  void Train(std::vector<Vector> &inputs, std::vector<Vector> &targets, size_t epochs, size_t batch_size, NeuralOptimizer &optimizer) {
    PrepareBatches(inputs, targets, batch_size);
    for (size_t i = 0; i < epochs; i++) {
      SampleBatches();
      double error = 0.0;
      for (size_t j = 0; j < batches_.size(); j++) {
        ZeroGradient();
        for (size_t k = 0; k < batches_[j].size(); k++) {
          ForwardPass(inputs[batches_[j][k]]);
          BackwardPass(targets[batches_[j][k]]);
        }
        ScaleGradient(batches_[j].size());
        optimizer.Step(w_, dw_, b_, db_);
        error += Error();
      }
      printf("epoch %ld - training error: %e\n", i, error / inputs.size());
    }
  }

private:
  double rms_;
  size_t n_;
  Vector i_;
  Vector e_;
  std::vector<size_t> dataset_indices_;
  std::vector<std::vector<size_t>> batches_;
  size_t batches_count_;
  size_t batches_residual_;
  size_t major_batch_size_;
  std::vector<Vector> d_;
  std::vector<Vector> z_;
  std::vector<Vector> a_;
  std::vector<Vector> p_;
  std::vector<Vector> b_;
  std::vector<Vector> db_;
  std::vector<Matrix> w_;
  std::vector<Matrix> dw_;
};
