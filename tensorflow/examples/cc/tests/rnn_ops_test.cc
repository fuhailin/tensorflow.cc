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
#define VOCAB_SIZE 8 // "helo wrd"

// #define VERBOSE 1
// #define TESTING 1

// VanillaRNN node builder 
static tensorflow::Status VanillaRNN(const tensorflow::Scope& scope, 
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

// VanillaRNNGrad node builder
static tensorflow::Status VanillaRNNGrad(const tensorflow::Scope& scope, 
                                          tensorflow::Input x,
                                          tensorflow::Input y, 
                                          tensorflow::Input p, 
                                          tensorflow::Input h, 
                                          tensorflow::Input w_hh, 
                                          tensorflow::Input w_hy, 
                                          tensorflow::Input h_prev, 
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

  auto _p = tensorflow::ops::AsNodeOut(scope, p);
  if (!scope.ok()) 
    return scope.status();

  auto _h = tensorflow::ops::AsNodeOut(scope, h);
  if (!scope.ok()) 
    return scope.status();

  auto _w_hh = tensorflow::ops::AsNodeOut(scope, w_hh);
  if (!scope.ok()) 
    return scope.status();

  auto _w_hy = tensorflow::ops::AsNodeOut(scope, w_hy);
  if (!scope.ok()) 
    return scope.status();

  auto _h_prev = tensorflow::ops::AsNodeOut(scope, h_prev);
  if (!scope.ok()) 
    return scope.status();

  // Build node
  tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("VanillaRNNGrad");
  auto builder = tensorflow::NodeBuilder(unique_name, "VanillaRNNGrad")
                    .Input(_x)
                    .Input(_y)
                    .Input(_p)
                    .Input(_h)
                    .Input(_w_hh)
                    .Input(_w_hy)
                    .Input(_h_prev)
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


// ApplyAdagradTrick node builder
static tensorflow::Status ApplyAdagradTrick(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input var, ::tensorflow::Input accum,
                           ::tensorflow::Input lr, ::tensorflow::Input grad, 
                                          tensorflow::Output &output) {
  if (!scope.ok()) return scope.status();

  auto _var = ::tensorflow::ops::AsNodeOut(scope, var);
  if (!scope.ok()) return scope.status();
  auto _accum = ::tensorflow::ops::AsNodeOut(scope, accum);
  if (!scope.ok()) return scope.status();
  auto _lr = ::tensorflow::ops::AsNodeOut(scope, lr);
  if (!scope.ok()) return scope.status();
  auto _grad = ::tensorflow::ops::AsNodeOut(scope, grad);
  if (!scope.ok()) return scope.status();

  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("ApplyAdagradTrick");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "ApplyAdagradTrick")
                     .Input(_var)
                     .Input(_accum)
                     .Input(_lr)
                     .Input(_grad)
                     .Attr("use_locking", false)
                     .Attr("update_slots", true)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return scope.status();
  scope.UpdateStatus(scope.DoShapeInference(ret));

  output = tensorflow::Output(ret, 0);

  return tensorflow::Status::OK();
}


int main() {

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();

  // vocabulary vs index
  const std::map<int, char> index_vocab_dic = {{0, ' '}, {1, 'e'}, {2, 'd'}, {3, 'h'}, {4, 'l'}, {5, 'o'}, {6, 'r'}, {7, 'w'}};
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
  h_prev_t.setZero();
#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "----------------h_prev_t: " << std::endl << h_prev_t;  
#endif

  // Trainable parameters start here, to be improved
  auto w_xh = tensorflow::ops::Variable(root, {HIDDEN_SIZE, VOCAB_SIZE}, tensorflow::DT_FLOAT);
  auto rate = tensorflow::ops::Const(root, {0.01f});
  auto random_value = tensorflow::ops::RandomNormal(root, {HIDDEN_SIZE, VOCAB_SIZE}, tensorflow::DT_FLOAT);
  auto assign_w_xh = tensorflow::ops::Assign(root, w_xh, tensorflow::ops::Multiply(root, random_value, rate));

  auto w_hh = tensorflow::ops::Variable(root, {HIDDEN_SIZE, HIDDEN_SIZE}, tensorflow::DT_FLOAT);
  auto random_value2 = tensorflow::ops::RandomNormal(root, {HIDDEN_SIZE, HIDDEN_SIZE}, tensorflow::DT_FLOAT);
  auto assign_w_hh = tensorflow::ops::Assign(root, w_hh, tensorflow::ops::Multiply(root, random_value2, rate));

  auto w_hy = tensorflow::ops::Variable(root, {VOCAB_SIZE, HIDDEN_SIZE}, tensorflow::DT_FLOAT);
  auto random_value3 = tensorflow::ops::RandomNormal(root, {VOCAB_SIZE, HIDDEN_SIZE}, tensorflow::DT_FLOAT);
  auto assign_w_hy = tensorflow::ops::Assign(root, w_hy, tensorflow::ops::Multiply(root, random_value3, rate));

  auto b_h = tensorflow::ops::Variable(root, {HIDDEN_SIZE, 1}, tensorflow::DT_FLOAT);
  tensorflow::Tensor b_h_zero_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({HIDDEN_SIZE, 1}));
  b_h_zero_tensor.matrix<float>().setZero();
  auto assign_b_h = tensorflow::ops::Assign(root, b_h, tensorflow::ops::ZerosLike(root, b_h_zero_tensor));

  auto b_y = tensorflow::ops::Variable(root, {VOCAB_SIZE, 1}, tensorflow::DT_FLOAT);
  tensorflow::Tensor b_y_zero_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({VOCAB_SIZE, 1}));
  b_y_zero_tensor.matrix<float>().setZero();  
  auto assign_b_y = tensorflow::ops::Assign(root, b_y, tensorflow::ops::ZerosLike(root, b_y_zero_tensor));
  // Trainable parameters end here 

  // Gradient accum parameters start here
  auto ada_w_xh = tensorflow::ops::Variable(root, {HIDDEN_SIZE, VOCAB_SIZE}, tensorflow::DT_FLOAT);
  auto assign_ada_w_xh = tensorflow::ops::Assign(root, ada_w_xh, tensorflow::ops::ZerosLike(root, w_xh));

  auto ada_w_hh = tensorflow::ops::Variable(root, {HIDDEN_SIZE, HIDDEN_SIZE}, tensorflow::DT_FLOAT);
  auto assign_ada_w_hh = tensorflow::ops::Assign(root, ada_w_hh, tensorflow::ops::ZerosLike(root, w_hh));

  auto ada_w_hy = tensorflow::ops::Variable(root, {VOCAB_SIZE, HIDDEN_SIZE}, tensorflow::DT_FLOAT);
  auto assign_ada_w_hy = tensorflow::ops::Assign(root, ada_w_hy, tensorflow::ops::ZerosLike(root, w_hy));

  auto ada_b_h = tensorflow::ops::Variable(root, {HIDDEN_SIZE, 1}, tensorflow::DT_FLOAT);
  auto assign_ada_b_h = tensorflow::ops::Assign(root, ada_b_h, tensorflow::ops::ZerosLike(root, b_h));

  auto ada_b_y = tensorflow::ops::Variable(root, {VOCAB_SIZE, 1}, tensorflow::DT_FLOAT);
  auto assign_ada_b_y = tensorflow::ops::Assign(root, ada_b_y, tensorflow::ops::ZerosLike(root, b_y));

  // Gradient accum parameters end here 

  // Placeholders
  auto x = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({SEQ_LENGTH, VOCAB_SIZE, 1}));
  auto y = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({VOCAB_SIZE, 1}));
  auto h_prev = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({HIDDEN_SIZE, 1}));

  // VanillaRNN Node
  tensorflow::Output vanilla_rnn_output;
  if (!VanillaRNN(root, x, y, h_prev, w_xh, w_hh, w_hy, b_h, b_y, vanilla_rnn_output).ok()) {
    LOG(ERROR) << "-----------------------------------------status: " << root.status();
    return root.status().code();
  }

  // VanillaRNNGrad Node
  tensorflow::Output vanilla_rnn_grad_output;
  if (!VanillaRNNGrad(root, x, y, tensorflow::Output(vanilla_rnn_output.node(), 0), tensorflow::Output(vanilla_rnn_output.node(), 1), w_hh, w_hy, h_prev, vanilla_rnn_grad_output).ok()) {
    LOG(ERROR) << "-----------------------------------------status: " << root.status();
    return root.status().code();
  }

  // Gradient
  auto lr = tensorflow::ops::Cast(root, 0.01, tensorflow::DT_FLOAT);

  // alternative of tensorflow::ops::ApplyAdagrad
  tensorflow::Output  apply_w_xh;
  ApplyAdagradTrick(root, w_xh, ada_w_xh, lr, tensorflow::Output(vanilla_rnn_grad_output.node(), 0), apply_w_xh);
  tensorflow::Output  apply_w_hh;
  ApplyAdagradTrick(root, w_hh, ada_w_hh, lr, tensorflow::Output(vanilla_rnn_grad_output.node(), 1), apply_w_hh);
  tensorflow::Output  apply_w_hy;
  ApplyAdagradTrick(root, w_hy, ada_w_hy, lr, tensorflow::Output(vanilla_rnn_grad_output.node(), 2), apply_w_hy);
  tensorflow::Output  apply_b_h;
  ApplyAdagradTrick(root, b_h, ada_b_h, lr, tensorflow::Output(vanilla_rnn_grad_output.node(), 3), apply_b_h);
  tensorflow::Output  apply_b_y;
  ApplyAdagradTrick(root, b_y, ada_b_y, lr, tensorflow::Output(vanilla_rnn_grad_output.node(), 4), apply_b_y);

  std::vector<tensorflow::Tensor> outputs;
  tensorflow::ClientSession session(root);

  // Initialize variables
  TF_CHECK_OK(session.Run({assign_w_xh, assign_w_hh, assign_w_hy, assign_b_h, assign_b_y}, 
                          nullptr));
  TF_CHECK_OK(session.Run({assign_ada_w_xh, assign_ada_w_hh, assign_ada_w_hy, assign_ada_b_h, assign_ada_b_y}, 
                          nullptr));

  // Train and eval
  for(int step = 0; step < 10000; step++) {
    // Train

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
      m(0, vocab_index) = 1.0f;

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
    y_t(vocab_index, 0) = 1.0f;
#ifdef VERBOSE  
    LOG(INFO) << __FUNCTION__ << "----------------y_t: " << std::endl << y_t;  
#endif

    // Run 
    TF_CHECK_OK(session.Run({{x, x_tensor}, {y, y_tensor}, {h_prev, h_prev_tensor}}, 
                            {tensorflow::Output(vanilla_rnn_grad_output.node(), 0), tensorflow::Output(vanilla_rnn_grad_output.node(), 1), 
                              tensorflow::Output(vanilla_rnn_grad_output.node(), 2), tensorflow::Output(vanilla_rnn_grad_output.node(), 3), 
                              tensorflow::Output(vanilla_rnn_grad_output.node(), 4), tensorflow::Output(vanilla_rnn_output.node(), 1), tensorflow::Output(vanilla_rnn_output.node(), 2),
                              apply_w_xh, apply_w_hh, apply_w_hy, apply_b_h, apply_b_y}, 
                            {}, 
                            &outputs));
#ifdef VERBOSE  
    LOG(INFO) << "Print output: " << outputs[0].shape() << ", " << outputs[1].shape() << ", " << outputs[2].shape() 
                                    << ", " << outputs[3].shape() << ", " << outputs[4].shape()
                                    << ", " << outputs[5].shape() << ", " << outputs[6].shape();
    LOG(INFO) << "Print step: " << step << ", output: " << outputs[0].DebugString() << ", " << outputs[1].DebugString() << ", " << outputs[2].DebugString() 
                                    << ", " << outputs[3].DebugString() << ", " << outputs[4].DebugString()
                                    << ", " << outputs[5].DebugString() << ", " << outputs[6].DebugString();
#endif
    LOG(INFO) << "Print step: " << step << ", loss: " << outputs[6].DebugString();

    // Update h_prev
    CHECK(h_prev_tensor.CopyFrom(outputs[5].Slice(SEQ_LENGTH - 1, SEQ_LENGTH), {outputs[5].dim_size(1), outputs[5].dim_size(2)}));
#ifdef VERBOSE  
    LOG(INFO) << __FUNCTION__ << "----------------------------h_prev_tensor updated:" << std::endl << h_prev_tensor.matrix<float>();
#endif

    // Evaluate



    // start over
    if(content_index > 350)
      content_index = 0;
  }

  return 0;
}