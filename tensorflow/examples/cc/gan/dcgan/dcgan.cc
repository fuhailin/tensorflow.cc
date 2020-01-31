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
#include "tensorflow/examples/cc/gan/dcgan/optimizer.h"
#include "tensorflow/examples/cc/gan/dcgan/util.h"

#ifdef ENABLE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#endif

using namespace tensorflow;                 // NOLINT(build/namespaces)
using namespace tensorflow::ops;            // NOLINT(build/namespaces)
using namespace tensorflow::ops::internal;  // NOLINT(build/namespaces)
using namespace std;                        // NOLINT(build/namespaces)

std::string DetailedDebugString(const Tensor& tensor) {
  return strings::StrCat("Tensor<type: ", DataTypeString(tensor.dtype()),
                         " shape: ", tensor.shape().DebugString(),
                         " values: ", tensor.SummarizeValue(-1, true), ">");
}

#ifdef TESTING

void test(const Scope& scope) {
  // Test SigmoidCrossEntropyWithLogits
  {
    // cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    // logits = tf.constant([[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    // labels = tf.ones_like(logits)
    // ce = cross_entropy(labels, logits)
    // print('ce: ', ce)

    auto logits = Const(scope, {{-1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}});
    auto ones_like = OnesLike(scope, logits);
    auto scel = SigmoidCrossEntropyWithLogits(scope, ones_like, logits);
    auto result = ReduceMean(scope, scel, {0, 1});

    vector<Tensor> outputs;
    ClientSession session(scope);

    Status status = session.Run({}, {result}, {}, &outputs);

    LOG(INFO) << "Print: SigmoidCrossEntropyWithLogits result: "
              << outputs[0].DebugString();
  }

  // Test BatchNormalization
  {
    auto x = Const<float>(scope, {{-1.0, 2.0, 1.0}, {1.0, 1.0, 1.0}});
    auto mean = Const<float>(scope, {0.0f});
    auto variance = Const<float>(scope, {1.0f});
    auto offset = Const<float>(scope, {0.0f});
    auto scale = Const<float>(scope, {1.0f});
    auto variance_epsilon = Const<float>(scope, {0.001f});
    auto batchnorm = BatchNormalization(scope, x, mean, variance, offset, scale,
                                        variance_epsilon);

    vector<Tensor> outputs;
    ClientSession session(scope);

    Status status = session.Run({}, {batchnorm}, {}, &outputs);

    LOG(INFO) << "Print: BatchNormalization result: "
              << DetailedDebugString(outputs[0]);
  }

  // Test Dropout
  {
    auto x = Const<float>(scope, {{-1.0, 2.0, 1.0}, {1.0, 1.0, 1.0}});
    auto dropout = Dropout(scope, x, {0.3f});

    vector<Tensor> outputs;
    ClientSession session(scope);

    Status status = session.Run({}, {dropout}, {}, &outputs);

    LOG(INFO) << "Print: Dropout result: " << DetailedDebugString(outputs[0]);
  }

  // Test GlorotUniform
  {
    auto glorot_uniform = GlorotUniform(scope, {3, 4});

    vector<Tensor> outputs;
    ClientSession session(scope);

    Status status = session.Run({}, {glorot_uniform}, {}, &outputs);

    LOG(INFO) << "Print: GlorotUniform result: "
              << DetailedDebugString(outputs[0]);
  }

  // Test Conv2DTranspose
  {
    int batch_size = 1;

    // Const
    auto const1 = Const<float>(scope, 1.0f, {batch_size, 7, 7, 256});
    LOG(INFO) << "Node building status: " << scope.status();

    // Conv2DTranspose 1
    auto input_sizes = Const<int>(scope, {batch_size, 7, 7, 128});
    // filter, aka kernel
    auto filter = Variable(scope, {5, 5, 128, 256}, DT_FLOAT);
    auto random_value1 = GlorotUniform(scope, {5, 5, 128, 256});
    auto assign_filter = Assign(scope, filter, random_value1);

    // out_backprop, aka input. here it's reshape1
    auto deconv1 = Conv2DTranspose(scope, input_sizes, filter, const1,
                                   {1, 1, 1, 1}, "SAME");
    LOG(INFO) << "Node building status: " << scope.status();

    vector<Tensor> outputs;
    ClientSession session(scope);

    TF_CHECK_OK(session.Run({assign_filter}, nullptr));
    TF_CHECK_OK(session.Run({}, {deconv1}, {}, &outputs));

    LOG(INFO) << "Print: Conv2DTranspose result: "
              << DetailedDebugString(outputs[0]);
  }
}

#endif

static Output DiscriminatorLoss(const Scope& scope, const Input& real_output,
                                const Input& fake_output) {
  auto real_loss =
      ReduceMean(scope,
                 SigmoidCrossEntropyWithLogits(
                     scope, OnesLike(scope, real_output), real_output),
                 {0, 1});
  auto fake_loss =
      ReduceMean(scope,
                 SigmoidCrossEntropyWithLogits(
                     scope, ZerosLike(scope, fake_output), fake_output),
                 {0, 1});

  return Add(scope, real_loss, fake_loss);
}

static Output GeneratorLoss(const Scope& scope, const Input& fake_output) {
  auto fake_loss =
      ReduceMean(scope,
                 SigmoidCrossEntropyWithLogits(
                     scope, OnesLike(scope, fake_output), fake_output),
                 {0, 1});

  return fake_loss;
}

int main() {
  Scope scope = Scope::NewRootScope();

#ifdef TESTING
  test(scope);
#endif

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

#ifdef VERBOSE
  // Session
  // Note that ClientSession can extend graph before running, Session cannot.
  vector<Tensor> dataset_outputs;

  // Run make_iterator_output first
  TF_CHECK_OK(session.Run({}, {}, {make_iterator_op}, nullptr));
  while (session.Run({}, iterator_get_next.components, &dataset_outputs).ok()) {
    LOG(INFO) << "Print dataset_outputs: " << dataset_outputs[0].DebugString();
  }
#endif

  //
  // generator and discriminator
  //

  //
  // Models
  auto generator = Generator(scope);
  auto discriminator = Discriminator(scope);

  // For inference
  const int num_examples_to_generate = 16;
  auto generator_inference =
      generator.Build(scope, num_examples_to_generate, false);
  auto discriminator_inference =
      discriminator.Build(scope, generator_inference, num_examples_to_generate);

  session.InitializeVariables(scope);

  // Run Test
  TF_CHECK_OK(session.Run({}, {generator_inference, discriminator_inference},
                          &outputs));
  LOG(INFO) << "Print discriminator output 0: " << outputs[0].DebugString();
  LOG(INFO) << "Print discriminator output 1: " << outputs[1].DebugString();

  // Train models
  auto generated_images = generator.Build(scope, BATCH_SIZE, true);
  auto real_output =
      discriminator.Build(scope, iterator_get_next.components[0], BATCH_SIZE);
  auto fake_output = discriminator.Build(scope, generated_images, BATCH_SIZE);

  // Loss
  auto gen_loss = GeneratorLoss(scope, fake_output);
  auto disc_loss = DiscriminatorLoss(scope, real_output, fake_output);

  // trainable variables
  std::vector<Output> trainable_variables_gen;
  scope.GetTrainableVariables({generated_images.node()->name()}, {},
                              &trainable_variables_gen);

  std::vector<Output> trainable_variables_disc;
  scope.GetTrainableVariables({real_output.node()->name()},
                              {generated_images.node()->name()},
                              &trainable_variables_disc);

#ifdef VERBOSE
  LOG(INFO) << "Training node name : " << generated_images.node()->name();

  for (auto iter = trainable_variables_gen.begin();
       iter != trainable_variables_gen.end(); ++iter) {
    LOG(INFO) << "trainable_variables_gen iter " << iter->name();
  }

  LOG(INFO) << "Training node name : " << real_output.node()->name() << " - "
            << generated_images.node()->name();

  for (auto iter = trainable_variables_disc.begin();
       iter != trainable_variables_disc.end(); ++iter) {
    LOG(INFO) << "trainable_variables_disc iter " << iter->name();
  }
#endif

  // Optimization
  AdamOptimizer adam_optimizer(scope);
  adam_optimizer.Build(scope, {gen_loss}, trainable_variables_gen);
  adam_optimizer.Build(scope, {disc_loss}, trainable_variables_disc);

  // Initialize variables
  session.InitializeVariables(scope);

  // Train
  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    // Reset dataset
    TF_CHECK_OK(session.Run({}, {}, {make_iterator_op}, nullptr));

    // batches training
    while (true) {
      vector<Tensor> outputs;

      // Run adam optimizer
      Status status = adam_optimizer.Run(scope, session, &outputs);
      if (status.ok()) {
        LOG(INFO) << "Print epoch: " << epoch
                  << ", gen_loss: " << outputs[0].DebugString();
        LOG(INFO) << "Print epoch: " << epoch
                  << ", disc_loss: " << outputs[1].DebugString();
      } else {
        // OUT_OF_RANGE means reaching the end of the dataset
        if (status.code() != tensorflow::error::OUT_OF_RANGE)
          LOG(INFO) << "Print epoch: " << epoch << ", status: " << status;

        break;
      }

      // Run update ops
      session.RunUpdateOps(scope);
#ifdef VERBOSE
      auto update_ops = scope.GetUpdateOps();
      for (auto& op : *update_ops) {
        LOG(INFO) << "GetUpdateOps op: " << op.node()->name();
      }
#endif
    }

#ifdef ENABLE_OPENCV
    // Run Inference
    TF_CHECK_OK(session.Run({}, {generator_inference}, &outputs));
    LOG(INFO) << "Inference generate output 0: " << outputs[0].DebugString();

    Tensor predict = outputs[0];
    for (int index = 0; index < num_examples_to_generate; index++) {
      Tensor image_data = predict.SubSlice(index);

      cv::Mat mat(IMAGE_SIZE, IMAGE_SIZE, CV_32F,
                  image_data.flat<float>().data());
      const float HALF_PIXEL_DEPTH = PIXEL_DEPTH / 2.0f;
      mat = mat * HALF_PIXEL_DEPTH + HALF_PIXEL_DEPTH;

      // Save images in folder "/tmp/data/"
      std::string filename = "/tmp/data/predict_" + std::to_string(epoch) +
                             "_" + std::to_string(index) + ".png";
      cv::imwrite(filename, mat);
    }
#endif
  }

  return 0;
}
