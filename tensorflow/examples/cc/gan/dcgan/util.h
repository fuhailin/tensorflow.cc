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

#ifndef TENSORFLOW_EXAMPLES_CC_GAN_DCGAN_UTIL_H_
#define TENSORFLOW_EXAMPLES_CC_GAN_DCGAN_UTIL_H_

// #define VERBOSE 1
// #define TESTING 1

// Adjustable Parameters
#define VALIDATION_SIZE 5000  // Size of the validation set.
#define SEED 66478            // Set to None for random seed.
#define BATCH_SIZE 256
#define BATCHES_PER_EPOCHS 214  // (55000 / BATCH_SIZE) = 214 max
#define NUM_EPOCHS 50
#define EVAL_BATCH_SIZE 256
#define EVAL_FREQUENCY 100  // Number of steps between evaluations.
#define LEARNING_RATE 0.0001f
#define BETA_1 0.9f
#define BETA_2 0.999f
#define EPSILON 1e-7

// Not to change
#define IMAGE_SIZE 28
#define NUM_CHANNELS 1
#define PIXEL_DEPTH 255.0f
#define NUM_LABELS 10
#define INPUTS_HEADER_BYTES 16
#define LABELS_HEADER_BYTES 8
#define NUM_IMAGES (BATCH_SIZE * BATCHES_PER_EPOCHS)  // 55000 images max
#define NOISE_DIM 100
#define UNITS (7 * 7 * 256)

//
// Python code: seed1, seed2 = random_seed.get_seed(seed)
//
static const int DEFAULT_GRAPH_SEED = 87654321;
static const int MAXINT32 = std::pow(2, 31) - 1;

int get_seed1(int seed) { return DEFAULT_GRAPH_SEED; }
int get_seed2(int seed) { return seed % MAXINT32; }

#endif  // TENSORFLOW_EXAMPLES_CC_GAN_DCGAN_UTIL_H_
