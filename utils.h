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
#include <cstdlib>
#include <cstdio>
#include <limits>

class RandomGenerator {
public:
  RandomGenerator();
  RandomGenerator(uint64_t seed);
  virtual ~RandomGenerator();
  void Seed(uint64_t seed);
  uint64_t Uint64();
  double Double();
  double Uniform();

private:
  uint64_t state_;
};

RandomGenerator::RandomGenerator() : state_(123456789) {}

RandomGenerator::RandomGenerator(uint64_t seed) : state_(seed) {}

RandomGenerator::~RandomGenerator() {}

void RandomGenerator::Seed(uint64_t seed) { state_ = seed; }

uint64_t RandomGenerator::Uint64() {
  state_ ^= state_ << 13;
  state_ ^= state_ >> 7;
  state_ ^= state_ << 17;
  return state_;
}

double RandomGenerator::Double() { return static_cast<double>(Uint64()); }

double RandomGenerator::Uniform() {
  return Double() / static_cast<double>(std::numeric_limits<uint64_t>::max());
}
