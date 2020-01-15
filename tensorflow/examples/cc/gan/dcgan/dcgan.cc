/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
// C++ implementation of DCGAN
// The correspoding python version:
// https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb
//
// Author: Rock Zhuang
// Date  : Jan 15, 2020
//

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/dataset_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/examples/cc/gan/dcgan/nn_ops_rkz.h"
#include "tensorflow/examples/cc/gan/dcgan/util.h"

using namespace tensorflow;                 // NOLINT(build/namespaces)
using namespace tensorflow::ops;            // NOLINT(build/namespaces)
using namespace tensorflow::ops::internal;  // NOLINT(build/namespaces)
using namespace std;                        // NOLINT(build/namespaces)

std::string DetailedDebugString(const Tensor& tensor) {
  return strings::StrCat("Tensor<type: ", DataTypeString(tensor.dtype()),
                         " shape: ", tensor.shape().DebugString(),
                         " values: ", tensor.SummarizeValue(-1, true), ">");
}

int main() {
  Scope scope = Scope::NewRootScope();

  //
  // Parse images files into tensors
  //

  // Read file and decompress data
  auto inputs_contents = ReadFile(
      scope, Const<string>(scope, "/tmp/data/train-images-idx3-ubyte.gz",
                           TensorShape({})));
  auto inputs_decode_compressed = DecodeCompressed(
      scope, inputs_contents, DecodeCompressed::CompressionType("GZIP"));

  vector<Tensor> outputs;
  ClientSession session(scope);

  // Load data into tensors
  Tensor inputs(DT_FLOAT, TensorShape({NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE,
                                       NUM_CHANNELS}));

  Status status = session.Run({}, {inputs_decode_compressed}, {}, &outputs);
  if (status.ok()) {
    // inputs
    std::string inputs_str = outputs[0].scalar<tstring>()();
    const char* inputs_str_data = inputs_str.c_str();

    float* inputs_data = inputs.tensor<float, 4>().data();
    int count = NUM_IMAGES * IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS;
    for (int i = 0; i < count; i++) {
      float data =
          (unsigned char)(*(inputs_str_data + INPUTS_HEADER_BYTES + i));

      // Normalize the images to [-1, 1]
      const float HALF_PIXEL_DEPTH = PIXEL_DEPTH / 2.0f;
      data = (data - HALF_PIXEL_DEPTH) / HALF_PIXEL_DEPTH;

      inputs_data[i] = data;
    }
  } else {
    LOG(INFO) << "Print: status: " << status;

    return -1;
  }

#ifdef VERBOSE
  LOG(INFO) << "Print: inputs: " << inputs.DebugString();
#endif

  // Convert tensors to TensorSliceDataset

  // Convert inputs into vector<Output>
  vector<Output> train_images;
  train_images.emplace_back(Const(scope, Input::Initializer(inputs)));

  // Prepare output_shapes
  vector<PartialTensorShape> output_shapes;
  output_shapes.emplace_back(
      tensorflow::PartialTensorShape({IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS}));

  // TensorSliceDataset
  auto tensor_slice_dataset =
      TensorSliceDataset(scope, train_images, output_shapes);

  // Shuffle and batch
  auto shuffle_dataset = ShuffleDataset(
      scope, tensor_slice_dataset,
      Cast(scope, NUM_IMAGES, DT_INT64),                   // buffer_size
      Cast(scope, 0, DT_INT64), Cast(scope, 0, DT_INT64),  // seedX
      std::initializer_list<DataType>{DT_FLOAT},           // output_types
      std::initializer_list<PartialTensorShape>{{}});      // output_shapes
  auto batch_dataset = BatchDataset(
      scope, shuffle_dataset, Cast(scope, BATCH_SIZE, DT_INT64),  // batch_size
      std::initializer_list<DataType>{DT_FLOAT},       // output_types
      std::initializer_list<PartialTensorShape>{{}});  // output_shapes

  // Iterator
  Output iterator_output =
      Iterator(scope, "iterator1", "", vector<DataType>({DT_FLOAT}),
               vector<PartialTensorShape>({{}}));
  Operation make_iterator_op =
      MakeIterator(scope, batch_dataset, iterator_output);
  auto iterator_get_next =
      IteratorGetNext(scope, iterator_output, vector<DataType>({DT_FLOAT}),
                      vector<PartialTensorShape>({{}}));

  // Session
  // Note that ClientSession can extend graph before running, Session cannot.
  vector<Tensor> dataset_outputs;

  // Run make_iterator_output first
  TF_CHECK_OK(session.Run({}, {}, {make_iterator_op}, nullptr));

#ifdef VERBOSE
  while (session.Run({}, iterator_get_next.components, &dataset_outputs).ok()) {
    LOG(INFO) << "Print dataset_outputs: " << dataset_outputs[0].DebugString();
  }
#endif

  //
  // generator and discriminator
  //

  // Test models
  auto generator = Generator(scope, 1);
  auto discriminator = Discriminator(scope, 1);

  // Initialize variables
  TF_CHECK_OK(session.Run({generator.assign_w1, generator.assign_filter,
                           generator.assign_filter2, generator.assign_filter3},
                          nullptr));
  TF_CHECK_OK(session.Run(
      {discriminator.assign_conv1_weights, discriminator.assign_conv1_biases,
       discriminator.assign_conv2_weights, discriminator.assign_conv2_biases,
       discriminator.assign_fc1_weights, discriminator.assign_fc1_biases},
      nullptr));
  TF_CHECK_OK(session.Run({discriminator.assign_accum_conv1_weights,
                           discriminator.assign_accum_conv1_biases,
                           discriminator.assign_accum_conv2_weights,
                           discriminator.assign_accum_conv2_biases,
                           discriminator.assign_accum_fc1_weights,
                           discriminator.assign_accum_fc1_biases},
                          nullptr));

  // Run
  TF_CHECK_OK(session.Run({{}}, {generator}, &outputs));
  LOG(INFO) << "Print generator output 0: " << outputs[0].DebugString();

  TF_CHECK_OK(session.Run({{discriminator.ph_inputs, outputs[0]}},
                          {discriminator}, &outputs));
  LOG(INFO) << "Print discriminator output 0: " << outputs[0].DebugString();

  return 0;
}
