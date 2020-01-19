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

  // Test Generator
  {
    auto test_generator = Generator(scope, 2);
    auto test_discriminator = Discriminator(scope, test_generator, 2);

    vector<Tensor> outputs;
    ClientSession session(scope);

    // Initialize variables
    TF_CHECK_OK(session.Run(
        {test_generator.assign_w1, test_generator.assign_filter,
         test_generator.assign_filter2, test_generator.assign_filter3},
        nullptr));
    TF_CHECK_OK(session.Run({test_discriminator.assign_conv1_weights,
                             test_discriminator.assign_conv1_biases,
                             test_discriminator.assign_conv2_weights,
                             test_discriminator.assign_conv2_biases,
                             test_discriminator.assign_fc1_weights,
                             test_discriminator.assign_fc1_biases},
                            nullptr));
    TF_CHECK_OK(session.Run(
        {test_discriminator.assign_conv1_wm, test_discriminator.assign_conv1_wv,
         test_discriminator.assign_conv1_bm, test_discriminator.assign_conv1_bv,
         test_discriminator.assign_conv2_wm, test_discriminator.assign_conv2_wv,
         test_discriminator.assign_conv2_bm, test_discriminator.assign_conv2_bv,
         test_discriminator.assign_fc1_wm, test_discriminator.assign_fc1_wv,
         test_discriminator.assign_fc1_bm, test_discriminator.assign_fc1_bv},
        nullptr));

    // Run Test
    // TF_CHECK_OK(session.Run({{}}, {test_generator}, &outputs));
    // LOG(INFO) << "Print generator output 0: " << outputs[0].DebugString();

    TF_CHECK_OK(
        session.Run({}, {test_generator, test_discriminator}, &outputs));
    LOG(INFO) << "Print discriminator output 0: " << outputs[0].DebugString();
    LOG(INFO) << "Print discriminator output 1: " << outputs[1].DebugString();
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

  return 0;
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
  // Test models
  auto test_generator = Generator(scope, 1);
  auto test_discriminator = Discriminator(scope, test_generator, 1);

  // Initialize variables
  TF_CHECK_OK(session.Run(
      {test_generator.assign_w1, test_generator.assign_filter,
       test_generator.assign_filter2, test_generator.assign_filter3,
       test_generator.assign_w1_wm, test_generator.assign_w1_wv,
       test_generator.assign_filter_wm, test_generator.assign_filter_wv,
       test_generator.assign_filter2_wm, test_generator.assign_filter2_wv,
       test_generator.assign_filter3_wm, test_generator.assign_filter3_wv},
      nullptr));
  TF_CHECK_OK(session.Run({test_discriminator.assign_conv1_weights,
                           test_discriminator.assign_conv1_biases,
                           test_discriminator.assign_conv2_weights,
                           test_discriminator.assign_conv2_biases,
                           test_discriminator.assign_fc1_weights,
                           test_discriminator.assign_fc1_biases},
                          nullptr));
  TF_CHECK_OK(session.Run(
      {test_discriminator.assign_conv1_wm, test_discriminator.assign_conv1_wv,
       test_discriminator.assign_conv1_bm, test_discriminator.assign_conv1_bv,
       test_discriminator.assign_conv2_wm, test_discriminator.assign_conv2_wv,
       test_discriminator.assign_conv2_bm, test_discriminator.assign_conv2_bv,
       test_discriminator.assign_fc1_wm, test_discriminator.assign_fc1_wv,
       test_discriminator.assign_fc1_bm, test_discriminator.assign_fc1_bv},
      nullptr));

  // Run Test
  // TF_CHECK_OK(session.Run({{}}, {test_generator}, &outputs));
  // LOG(INFO) << "Print generator output 0: " << outputs[0].DebugString();

  TF_CHECK_OK(session.Run({}, {test_generator, test_discriminator}, &outputs));
  LOG(INFO) << "Print discriminator output 0: " << outputs[0].DebugString();
  LOG(INFO) << "Print discriminator output 1: " << outputs[1].DebugString();

  //
  // Train models
  auto generated_images = Generator(scope, BATCH_SIZE);
  auto real_output =
      Discriminator(scope, iterator_get_next.components[0], BATCH_SIZE);
  auto fake_output =
      Discriminator(scope, real_output, generated_images, BATCH_SIZE);

  // Loss
  auto gen_loss = GeneratorLoss(scope, fake_output);
  auto disc_loss = DiscriminatorLoss(scope, real_output, fake_output);

  // Gradient
  std::vector<Output> grad_outputs_gen;
  TF_CHECK_OK(
      AddSymbolicGradients(scope, {gen_loss},
                           {generated_images.w1, generated_images.filter,
                            generated_images.filter2, generated_images.filter3},
                           &grad_outputs_gen));
  LOG(INFO) << "Node building status: " << scope.status();

  std::vector<Output> grad_outputs_disc;
  TF_CHECK_OK(AddSymbolicGradients(
      scope, {disc_loss},
      {real_output.conv1_weights, real_output.conv2_weights,
       real_output.fc1_weights, real_output.conv1_biases,
       real_output.conv2_biases, real_output.fc1_biases},
      &grad_outputs_disc));
  LOG(INFO) << "Node building status: " << scope.status();

  // update the weights and bias using gradient descent
  // Use Adam
  auto lr = Const<float>(scope, LEARNING_RATE);
  auto beta1 = Const<float>(scope, BETA_1);
  auto beta2 = Const<float>(scope, BETA_2);
  auto epsilon = Const<float>(scope, EPSILON);

  auto global_step = Variable(scope, {}, DT_FLOAT);
  auto assign_global_step = Assign(scope, global_step, 1.0f);
  auto assign_add_global_step = AssignAdd(scope, global_step, 1.0f);

  auto beta1_power = Pow(scope, beta1, global_step);
  auto beta2_power = Pow(scope, beta2, global_step);

  // Generator
  auto apply_w1_gen =
      ApplyAdam(scope, generated_images.w1, generated_images.w1_wm,
                generated_images.w1_wv, beta1_power, beta2_power, lr, beta1,
                beta2, epsilon, grad_outputs_gen[0]);
  LOG(INFO) << "Node building status: " << scope.status();

  auto apply_filter_gen =
      ApplyAdam(scope, generated_images.filter, generated_images.filter_wm,
                generated_images.filter_wv, beta1_power, beta2_power, lr, beta1,
                beta2, epsilon, grad_outputs_gen[1]);
  LOG(INFO) << "Node building status: " << scope.status();

  auto apply_filter2_gen =
      ApplyAdam(scope, generated_images.filter2, generated_images.filter2_wm,
                generated_images.filter2_wv, beta1_power, beta2_power, lr,
                beta1, beta2, epsilon, grad_outputs_gen[2]);
  LOG(INFO) << "Node building status: " << scope.status();

  auto apply_filter3_gen =
      ApplyAdam(scope, generated_images.filter3, generated_images.filter3_wm,
                generated_images.filter3_wv, beta1_power, beta2_power, lr,
                beta1, beta2, epsilon, grad_outputs_gen[3]);
  LOG(INFO) << "Node building status: " << scope.status();

  // discriminator
  auto apply_conv1_weights_disc =
      ApplyAdam(scope, fake_output.conv1_weights, real_output.conv1_wm,
                real_output.conv1_wv, beta1_power, beta2_power, lr, beta1,
                beta2, epsilon, grad_outputs_disc[0]);
  LOG(INFO) << "Node building status: " << scope.status();

  auto apply_conv2_weights_disc =
      ApplyAdam(scope, real_output.conv2_weights, real_output.conv2_wm,
                real_output.conv2_wv, beta1_power, beta2_power, lr, beta1,
                beta2, epsilon, grad_outputs_disc[1]);
  LOG(INFO) << "Node building status: " << scope.status();

  auto apply_fc1_weights_disc =
      ApplyAdam(scope, real_output.fc1_weights, real_output.fc1_wm,
                real_output.fc1_wv, beta1_power, beta2_power, lr, beta1, beta2,
                epsilon, grad_outputs_disc[2]);
  LOG(INFO) << "Node building status: " << scope.status();

  auto apply_conv1_biases_disc =
      ApplyAdam(scope, real_output.conv1_biases, real_output.conv1_bm,
                real_output.conv1_bv, beta1_power, beta2_power, lr, beta1,
                beta2, epsilon, grad_outputs_disc[3]);
  LOG(INFO) << "Node building status: " << scope.status();

  auto apply_conv2_biases_disc =
      ApplyAdam(scope, real_output.conv2_biases, real_output.conv2_bm,
                real_output.conv2_bv, beta1_power, beta2_power, lr, beta1,
                beta2, epsilon, grad_outputs_disc[4]);
  LOG(INFO) << "Node building status: " << scope.status();

  auto apply_fc1_biases_disc =
      ApplyAdam(scope, real_output.fc1_biases, real_output.fc1_bm,
                real_output.fc1_bv, beta1_power, beta2_power, lr, beta1, beta2,
                epsilon, grad_outputs_disc[5]);
  LOG(INFO) << "Node building status: " << scope.status();

  // Initialize variables
  TF_CHECK_OK(session.Run({assign_global_step}, nullptr));
  TF_CHECK_OK(session.Run(
      {generated_images.assign_w1, generated_images.assign_filter,
       generated_images.assign_filter2, generated_images.assign_filter3,
       generated_images.assign_w1_wm, generated_images.assign_w1_wv,
       generated_images.assign_filter_wm, generated_images.assign_filter_wv,
       generated_images.assign_filter2_wm, generated_images.assign_filter2_wv,
       generated_images.assign_filter3_wm, generated_images.assign_filter3_wv},
      nullptr));
  TF_CHECK_OK(session.Run(
      {real_output.assign_conv1_weights, real_output.assign_conv1_biases,
       real_output.assign_conv2_weights, real_output.assign_conv2_biases,
       real_output.assign_fc1_weights, real_output.assign_fc1_biases,
       real_output.assign_conv1_wm, real_output.assign_conv1_wv,
       real_output.assign_conv1_bm, real_output.assign_conv1_bv,
       real_output.assign_conv2_wm, real_output.assign_conv2_wv,
       real_output.assign_conv2_bm, real_output.assign_conv2_bv,
       real_output.assign_fc1_wm, real_output.assign_fc1_wv,
       real_output.assign_fc1_bm, real_output.assign_fc1_bv},
      nullptr));

  // Train
  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    TF_CHECK_OK(session.Run({}, {}, {make_iterator_op}, nullptr));

    // batches training
    while (true) {
      vector<Tensor> outputs;
      Status status =
          session.Run({gen_loss, disc_loss, apply_w1_gen, apply_filter_gen,
                       apply_filter2_gen, apply_filter3_gen,
                       apply_conv1_weights_disc, apply_conv2_weights_disc,
                       apply_fc1_weights_disc, apply_conv1_biases_disc,
                       apply_conv2_biases_disc, apply_fc1_biases_disc},
                      &outputs);
      if (status.ok()) {
#ifdef VERBOSE
        LOG(INFO) << "Print epoch: " << epoch
                  << ", gen_loss: " << outputs[0].DebugString();
        LOG(INFO) << "Print epoch: " << epoch
                  << ", disc_loss: " << outputs[1].DebugString();
#endif
      } else {
        if (status.code() != tensorflow::error::OUT_OF_RANGE)
          LOG(INFO) << "Print epoch: " << epoch << ", status: " << status;

        break;
      }

      vector<Tensor> assign_add_outputs;
      TF_CHECK_OK(session.Run({assign_add_global_step}, &assign_add_outputs));

      int step = static_cast<int>(assign_add_outputs[0].scalar<float>()());
      if (step % EVAL_FREQUENCY == 0) {
        LOG(INFO) << "Print step: " << step << ", epoch: " << epoch
                  << ", gen_loss: " << outputs[0].DebugString()
                  << ", disc_loss: " << outputs[1].DebugString();
      }
    }
  }

  return 0;
}
