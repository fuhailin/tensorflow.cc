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

#ifndef TENSORFLOW_EXAMPLES_CC_MAGENTA_RL_TUNER_CONST_H_
#define TENSORFLOW_EXAMPLES_CC_MAGENTA_RL_TUNER_CONST_H_

// #define VERBOSE 1
// #define TESTING 1

// Adjustable parameters
#define NUM_UNIT 128             // HIDDEN_SIZE
#define TIME_LEN 384             // NUM_STEPS
#define BATCH_SIZE 10            // BATCH_SIZE
#define TRAINING_STEPS 10000
#define MINIBATCH_SIZE 32

// Don't change
#define INPUT_SIZE 38            // (DEFAULT_MAX_NOTE(84) - DEFAULT_MIN_NOTE(48) + NUM_SPECIAL_MELODY_EVENTS(2))
#define SEQ_LENGTH TIME_LEN * BATCH_SIZE

// #define LIBRARY_FILENAME "/home/rock/.cache/bazel/_bazel_rock/9982590d8d227cddee8c85cf45e44b89/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/contrib/rnn/python/ops/_lstm_ops.so"
#define LIBRARY_FILENAME "/../../../../../../tensorflow/contrib/rnn/python/ops/_lstm_ops.so"

#endif  // TENSORFLOW_EXAMPLES_CC_MAGENTA_RL_TUNER_CONST_H_
