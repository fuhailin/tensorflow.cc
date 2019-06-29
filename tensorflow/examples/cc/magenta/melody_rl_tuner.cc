/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//
// C++ implementation of Magenta rl_tuner module
// Using reinforcement learning with RNN(LSTM)
//
// Draft 2! Working in progress!!
//
// Code check supports: cpplint --filter=-build/namespaces --linelength=120 xxx
//
// Author: Rock Zhuang
// Date  : Jun 25, 2019
//

#include "tensorflow/examples/cc/magenta/rl_tuner/rl_tuner.h"

// #define LIBRARY_FILENAME "/home/rock/.cache/bazel/_bazel_rock/9982590d8d227cddee8c85cf45e44b89/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/contrib/rnn/python/ops/_lstm_ops.so"  // NOLINT
#define LIBRARY_FILENAME "/../../../../../../tensorflow/contrib/rnn/python/ops/_lstm_ops.so"

using tensorflow::RLTuner;
using tensorflow::Env;

std::string get_working_path() {
  char temp[MAXPATHLEN];
  return (getcwd(temp, sizeof(temp)) ? std::string(temp) : std::string(""));
}

int main() {
  // Load library of lstm_ops
  std::string path = get_working_path();
  void* unused_filehandle;
  TF_CHECK_OK(Env::Default()->LoadLibrary(path.append(LIBRARY_FILENAME).c_str(), &unused_filehandle));
  // TF_CHECK_OK(Env::Default()->LoadLibrary(LIBRARY_FILENAME, &unused_filehandle));

  // RLTuner
  RLTuner rlTuner;
  rlTuner.Train();

  return 0;
}
