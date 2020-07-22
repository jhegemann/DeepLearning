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

#include <cstdlib>
#include <cstdio>

#include "network.h"

int main(int argc, char **argv) {
  NeuralNetwork network;
  network.Init( {1, 32, 32, 1} );

  std::vector<Vector> inputs;
  std::vector<Vector> outputs;

  for (double x = -M_PI; x <= M_PI; x += 1.0e-2) {
    inputs.emplace_back(1);
    inputs.back()(0) = x;
    outputs.emplace_back(1);
    outputs.back()(0) = 0.5 + 0.25 * sin(x);
  }

  network.Train(inputs, outputs, 2500, inputs.size());

  std::fstream file;
  file.open("sin.txt", std::fstream::out);
  for (double x = -M_PI; x <= M_PI; x += 1.0e-2) {
    Vector input(1);
    input(0) = x;
    Vector output = network.Predict(input);
    file << x << " " << output(0) << std::endl;
  }
  file.close();

}
