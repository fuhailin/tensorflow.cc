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
// Example for using rnn ops in C++, Vanallia RNN for now
// Check vanilla_rnn.py for python numpy implementation
// Author: Rock Zhuang
// Date  : Dec 20, 2018
// 

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

// using namespace tensorflow;
// using namespace tensorflow::ops;

// Test sample
const char test_content[] = "hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world";
#define HIDDEN_SIZE 16
#define SEQ_LENGTH 10
#define VOCAB_SIZE 8 // "helo wrd" -> 1/2/3/4/5/6/7/8

#define VERBOSE 1
#define TESTING 1

// VanillaRNN node builder with Placeholder
static tensorflow::Status VanillaRNNPlaceholder(const tensorflow::Scope& scope, 
                                          tensorflow::Input x,
                                          tensorflow::Input y, 
                                          tensorflow::Input h_prev, 
                                          tensorflow::Input w_xh, 
                                          tensorflow::Input w_hh, 
                                          tensorflow::Input w_hy, 
                                          tensorflow::Input b_h, 
                                          tensorflow::Input b_y, 
                                          tensorflow::Output &output) {
  if (!scope.ok()) 
    return  scope.status();  
  
  // Prepare inputs
  auto _x = tensorflow::ops::AsNodeOut(scope, x);
  if (!scope.ok()) 
    return scope.status();

  auto _y = tensorflow::ops::AsNodeOut(scope, y);
  if (!scope.ok()) 
    return scope.status();

  auto _h_prev = tensorflow::ops::AsNodeOut(scope, h_prev);
  if (!scope.ok()) 
    return scope.status();

  auto _w_xh = tensorflow::ops::AsNodeOut(scope, w_xh);
  if (!scope.ok()) 
    return scope.status();

  auto _w_hh = tensorflow::ops::AsNodeOut(scope, w_hh);
  if (!scope.ok()) 
    return scope.status();

  auto _w_hy = tensorflow::ops::AsNodeOut(scope, w_hy);
  if (!scope.ok()) 
    return scope.status();

  auto _b_h = tensorflow::ops::AsNodeOut(scope, b_h);
  if (!scope.ok()) 
    return scope.status();

  auto _b_y = tensorflow::ops::AsNodeOut(scope, b_y);
  if (!scope.ok()) 
    return scope.status();

  // Build node
  tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("VanillaRNN");
  auto builder = tensorflow::NodeBuilder(unique_name, "VanillaRNN")
                    .Input(_x)
                    .Input(_y)
                    .Input(_h_prev)
                    .Input(_w_xh)
                    .Input(_w_hh)
                    .Input(_w_hy)
                    .Input(_b_h)
                    .Input(_b_y)
                    .Attr("hidsize", HIDDEN_SIZE);

  // Update scope
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));

  if (!scope.ok()) 
    return scope.status();

  scope.UpdateStatus(scope.DoShapeInference(ret));

  // Output
  output = tensorflow::Output(ret, 0);

  return tensorflow::Status::OK();
}

// VanillaRNN node with extra inputs
static tensorflow::Status VanillaRNN(const tensorflow::Scope& scope, 
                                          tensorflow::Input x,
                                          tensorflow::Input y, 
                                          tensorflow::Input h_prev, 
                                          tensorflow::Input w_xh, 
                                          tensorflow::Input w_hh, 
                                          tensorflow::Input w_hy, 
                                          tensorflow::Input b_h, 
                                          tensorflow::Input b_y, 
                                          const int hidsize,
                                          tensorflow::Output &output) {
  if (!scope.ok()) 
    return  scope.status();

  // Prepare inputs
  auto _x = tensorflow::ops::AsNodeOut(scope, x);
  if (!scope.ok()) 
    return scope.status();

  auto _y = tensorflow::ops::AsNodeOut(scope, y);
  if (!scope.ok()) 
    return scope.status();

  auto _h_prev = tensorflow::ops::AsNodeOut(scope, h_prev);
  if (!scope.ok()) 
    return scope.status();

  auto _w_xh = tensorflow::ops::AsNodeOut(scope, w_xh);
  if (!scope.ok()) 
    return scope.status();

  auto _w_hh = tensorflow::ops::AsNodeOut(scope, w_hh);
  if (!scope.ok()) 
    return scope.status();

  auto _w_hy = tensorflow::ops::AsNodeOut(scope, w_hy);
  if (!scope.ok()) 
    return scope.status();

  auto _b_h = tensorflow::ops::AsNodeOut(scope, b_h);
  if (!scope.ok()) 
    return scope.status();

  auto _b_y = tensorflow::ops::AsNodeOut(scope, b_y);
  if (!scope.ok()) 
    return scope.status();

  // Build node
  tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("VanillaRNN");
  auto builder = tensorflow::NodeBuilder(unique_name, "VanillaRNN")
                    .Input(_x)
                    .Input(_y)
                    .Input(_h_prev)
                    .Input(_w_xh)
                    .Input(_w_hh)
                    .Input(_w_hy)
                    .Input(_b_h)
                    .Input(_b_y)
                    .Attr("hidsize", HIDDEN_SIZE);

  // Update scope
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));

  if (!scope.ok()) 
    return scope.status();

  scope.UpdateStatus(scope.DoShapeInference(ret));

  // Output
  output = tensorflow::Output(ret, 0);

  return tensorflow::Status::OK();
}

int main() {

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();

  // vocabulary vs index
  const std::map<int, char> index_vocab_dic = {{1, 'h'}, {2, 'e'}, {3, 'l'}, {4, 'o'}, {5, ' '}, {6, 'w'}, {7, 'r'}, {8, 'd'}};
  std::map<char, int> vocab_index_dic;
  for(auto iter = index_vocab_dic.begin(); iter != index_vocab_dic.end(); iter++) {
    // LOG(INFO) << "----------------index_vocab_dic: " << iter->first <<' ' << iter->second;  

    vocab_index_dic.insert(std::pair<char, int>(iter->second, iter->first));
  }

#ifdef VERBOSE
  // Check vocab_index_dic for fun
  for(auto iter = vocab_index_dic.begin(); iter != vocab_index_dic.end(); iter++) {
    LOG(INFO) << __FUNCTION__ << "----------------vocab_index_dic: " << iter->first <<' ' << iter->second;  
  }
#endif

  // Global content index
  int content_index = 0;

  // Prepare hidden tensor
  tensorflow::Tensor h_prev_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({HIDDEN_SIZE, 1}));
  typename tensorflow::TTypes<float>::Matrix h_prev_t = h_prev_tensor.matrix<float>();
  h_prev_t.setRandom();
#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "----------------h_prev_t: " << std::endl << h_prev_t;  
#endif

  // Trainable parameters start here 
  // Prepare w_xh tensor
  tensorflow::Tensor w_xh_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({HIDDEN_SIZE, VOCAB_SIZE}));
  typename tensorflow::TTypes<float>::Matrix w_xh_t = w_xh_tensor.matrix<float>();
  w_xh_t.setRandom();

  // Prepare w_hh tensor
  tensorflow::Tensor w_hh_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({HIDDEN_SIZE, HIDDEN_SIZE}));
  typename tensorflow::TTypes<float>::Matrix w_hh_t = w_hh_tensor.matrix<float>();
  w_hh_t.setRandom();

  // Prepare w_hy tensor
  tensorflow::Tensor w_hy_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({VOCAB_SIZE, HIDDEN_SIZE}));
  typename tensorflow::TTypes<float>::Matrix w_hy_t = w_hy_tensor.matrix<float>();
  w_hy_t.setRandom();

  // Prepare b_h tensor
  tensorflow::Tensor b_h_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({HIDDEN_SIZE, 1}));
  typename tensorflow::TTypes<float>::Matrix b_h_t = b_h_tensor.matrix<float>();
  b_h_t.setRandom();

  // Prepare b_y tensor
  tensorflow::Tensor b_y_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({VOCAB_SIZE, 1}));
  typename tensorflow::TTypes<float>::Matrix b_y_t = b_y_tensor.matrix<float>();
  b_y_t.setRandom();
  // Trainable parameters end here 


  // Placeholders
  auto x = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({SEQ_LENGTH, VOCAB_SIZE, 1}));
  auto y = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({SEQ_LENGTH, VOCAB_SIZE, 1}));
  auto h_prev = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({SEQ_LENGTH, VOCAB_SIZE, 1}));
  auto w_xh = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({SEQ_LENGTH, VOCAB_SIZE, 1}));
  auto w_hh = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({SEQ_LENGTH, VOCAB_SIZE, 1}));
  auto w_hy = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({SEQ_LENGTH, VOCAB_SIZE, 1}));
  auto b_h = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({SEQ_LENGTH, VOCAB_SIZE, 1}));
  auto b_y = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({SEQ_LENGTH, VOCAB_SIZE, 1}));

  // VanillaRNN Node
  tensorflow::Output vanilla_rnn_output;
  if (!VanillaRNNPlaceholder(root, x, y, h_prev, w_xh, w_hh, w_hy, b_h, b_y, vanilla_rnn_output).ok()) {
    LOG(ERROR) << "-----------------------------------------status: " << root.status();
    return root.status().code();
  }

  std::vector<tensorflow::Tensor> outputs;
  tensorflow::ClientSession session(root);

  // Train
  for(int step = 0; step < 10; step++) {
    // Forward propagation

    // Batch input with batch size of SEQ_LENGTH
    tensorflow::Tensor x_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({SEQ_LENGTH, VOCAB_SIZE, 1}));

    // batch one-hot processing
    auto e_2d = x_tensor.shaped<float, 2>({SEQ_LENGTH, VOCAB_SIZE * 1});
    for (int i = 0; i < SEQ_LENGTH; i++) {
      // Ref: tensor_test.cc

      // Assign a 1 x VOCAB_SIZE * 1 matrix (really vector) to a slice of size
      Eigen::Tensor<float, 2, Eigen::RowMajor> m(1, VOCAB_SIZE * 1);
      m.setZero();

      // one-hot processing for one character
      char test_char = test_content[content_index++];
      auto search = vocab_index_dic.find(test_char);
      int vocab_index = search->second;
      m(0, vocab_index - 1) = 1.0f;

      // set e_2d
      Eigen::DSizes<Eigen::DenseIndex, 2> indices(i, 0);
      Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, VOCAB_SIZE * 1);
      e_2d.slice(indices, sizes) = m;
    }

#ifdef VERBOSE
    // Check e_2d for fun
    LOG(INFO) << __FUNCTION__ << "----------------e_2d: " << std::endl << e_2d;  

    // Check x_tensor for fun
    auto e_t = x_tensor.tensor<float, 3>();
    LOG(INFO) << __FUNCTION__ << "----------------e_t: " << std::endl << e_t;  
#endif

    // Prepare y
    tensorflow::Tensor y_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({VOCAB_SIZE, 1}));
    typename tensorflow::TTypes<float>::Matrix y_t = y_tensor.matrix<float>();

    // Prepare y: one hot processing for one character
    y_t.setZero();
    char test_char = test_content[content_index];
    auto search = vocab_index_dic.find(test_char);
    int vocab_index = search->second;
    y_t(vocab_index - 1, 0) = 1.0f;
#ifdef VERBOSE  
    LOG(INFO) << __FUNCTION__ << "----------------y_t: " << std::endl << y_t;  
#endif

    // Run 
    TF_CHECK_OK(session.Run({{x, x_tensor}, {y, y_tensor}, {h_prev, h_prev_tensor}, {w_xh, w_xh_tensor}, {w_hh, w_hh_tensor}, {w_hy, w_hy_tensor}, {b_h, b_h_tensor}, {b_y, b_y_tensor} }, 
                            {tensorflow::Output(vanilla_rnn_output.node(), 0), tensorflow::Output(vanilla_rnn_output.node(), 1), tensorflow::Output(vanilla_rnn_output.node(), 2)}, 
                            {vanilla_rnn_output.op()}, 
                            &outputs));

    LOG(INFO) << "Print: " << outputs[0].shape() << ", " << outputs[1].shape() << ", " << outputs[2].shape();
    LOG(INFO) << "Print: " << outputs[0].DebugString() << ", " << outputs[1].DebugString() << ", " << outputs[2].DebugString();
  }

  // Backward propagation


  // Gradient



  return 0;
}