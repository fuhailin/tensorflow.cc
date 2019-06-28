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

#include "tensorflow/examples/cc/magenta/rl_tuner/const.h"
#include "tensorflow/examples/cc/magenta/rl_tuner/note_rnn.h"

using namespace tensorflow;
using namespace tensorflow::ops;
using namespace tensorflow::ops::internal;
using namespace std;

// #define LIBRARY_FILENAME "/home/rock/.cache/bazel/_bazel_rock/9982590d8d227cddee8c85cf45e44b89/execthis->scope/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/contrib/rnn/python/ops/_lstm_ops.so"
#define LIBRARY_FILENAME "/../../../../../../tensorflow/contrib/rnn/python/ops/_lstm_ops.so"

namespace tensorflow {

std::string get_working_path() {
  char temp[MAXPATHLEN];
  return (getcwd(temp, sizeof(temp)) ? std::string(temp) : std::string(""));
}

NoteRNN::NoteRNN(const ::tensorflow::Scope& s)
    : scope(s), session(scope) {
  std::string path = get_working_path();
  void* unused_filehandle;
  // TF_CHECK_OK(Env::Default()->LoadLibrary(LIBRARY_FILENAME, &unused_filehandle));
  TF_CHECK_OK(Env::Default()->LoadLibrary(path.append(LIBRARY_FILENAME).c_str(), &unused_filehandle));

  this->BuildGraph();
}

NoteRNN::~NoteRNN() {
}

Status NoteRNN::BuildGraph() {
  this->w = Variable(this->scope.WithOpName("w"), {INPUT_SIZE + NUM_UNIT, NUM_UNIT * 4}, DT_FLOAT);  // (input_size + cell_size, cell_size * 4) for {i, c, f, o}
  this->b = Variable(this->scope.WithOpName("b"), {NUM_UNIT * 4}, DT_FLOAT);
  this->w_y = Variable(this->scope.WithOpName("w_y"), {INPUT_SIZE, NUM_UNIT}, DT_FLOAT);
  this->b_y = Variable(this->scope.WithOpName("b_y"), {INPUT_SIZE}, DT_FLOAT);
  // this->cs = Variable(this->scope.WithOpName("cs"), {TIME_LEN, BATCH_SIZE, NUM_UNIT}, DT_FLOAT);

  this->ada_w = Variable(this->scope, {INPUT_SIZE + NUM_UNIT, NUM_UNIT * 4}, DT_FLOAT);
  this->ada_b = Variable(this->scope, {NUM_UNIT * 4}, DT_FLOAT);
  this->ada_w_y = Variable(this->scope, {INPUT_SIZE, NUM_UNIT}, DT_FLOAT);
  this->ada_b_y = Variable(this->scope, {INPUT_SIZE}, DT_FLOAT);

  this->x = Placeholder(this->scope, DT_FLOAT, Placeholder::Shape({TIME_LEN, BATCH_SIZE, INPUT_SIZE}));  // (timelen, batch_size, num_inputs)
  this->y = Placeholder(this->scope, DT_INT64, Placeholder::Shape({TIME_LEN, BATCH_SIZE}));  // (timelen, batch_size)

  this->cs_prev = Placeholder(this->scope, DT_FLOAT, Placeholder::Shape({BATCH_SIZE, NUM_UNIT}));  // (batch_size, cell_size)
  this->h_prev = Placeholder(this->scope, DT_FLOAT, Placeholder::Shape({BATCH_SIZE, NUM_UNIT}));

  // y_tensor not really used
  Tensor y_tensor(DT_INT64, TensorShape({TIME_LEN, BATCH_SIZE}));
  typename TTypes<int64>::Matrix y_tensor_t = y_tensor.matrix<int64>();
  y_tensor_t.setZero();

  // zeroed out when batch 0
  this->h_prev_tensor = Tensor(DT_FLOAT, TensorShape({BATCH_SIZE, NUM_UNIT}));
  typename TTypes<float>::Matrix h_prev_t = this->h_prev_tensor.matrix<float>();
  h_prev_t.setZero();

  this->cs_prev_tensor = Tensor(DT_FLOAT, TensorShape({BATCH_SIZE, NUM_UNIT}));
  typename TTypes<float>::Matrix cs_prev_t = this->cs_prev_tensor.matrix<float>();
  cs_prev_t.setZero();


  // LSTM
  this->block_lstm = std::shared_ptr<BlockLSTM>(new BlockLSTM(this->scope,
                              Const<int64>(this->scope, {TIME_LEN}),  // seq_len_max,
                              this->x,
                              this->cs_prev,
                              this->h_prev,
                              this->w,
                              Const<float>(this->scope, 0, TensorShape({NUM_UNIT})),  // wci, used when use_peephole is true
                              Const<float>(this->scope, 0, TensorShape({NUM_UNIT})),  // wcf
                              Const<float>(this->scope, 0, TensorShape({NUM_UNIT})),  // wco
                              this->b));
  LOG(INFO) << "Node building status: " << this->scope.status();

  this->rnn_softmax_loss = std::shared_ptr<RNNSoftmaxLoss>(new RNNSoftmaxLoss(this->scope,
                              this->block_lstm->h,
                              this->y,
                              this->w_y,
                              this->b_y,
                              this->block_lstm->cs));
  LOG(INFO) << "Node building status: " << this->scope.status();

  // TODO(Rock): rnn_softmax_loss->logits
  this->logits = Variable(this->scope.WithOpName("logits"), {INPUT_SIZE, MINIBATCH_SIZE}, DT_FLOAT);

  return this->scope.status();
}

Status NoteRNN::Init() {
  auto rate = Const(this->scope, {0.01f});
  auto random_value = RandomNormal(this->scope, {INPUT_SIZE + NUM_UNIT, NUM_UNIT * 4}, DT_FLOAT);
  auto assign_w = Assign(this->scope, w, Multiply(this->scope, random_value, rate));

  Tensor b_zero_tensor(DT_FLOAT, TensorShape({NUM_UNIT * 4}));
  b_zero_tensor.vec<float>().setZero();
  auto assign_b = Assign(this->scope, b, ZerosLike(this->scope, b_zero_tensor));

  auto random_value2 = RandomNormal(this->scope, {INPUT_SIZE, NUM_UNIT}, DT_FLOAT);
  auto assign_w_y = Assign(this->scope, w_y, Multiply(this->scope, random_value2, rate));

  Tensor b_y_zero_tensor(DT_FLOAT, TensorShape({INPUT_SIZE}));
  b_y_zero_tensor.vec<float>().setZero();
  auto assign_b_y = Assign(this->scope, b_y, ZerosLike(this->scope, b_y_zero_tensor));

  // Tensor cs_zero_tensor(DT_FLOAT, TensorShape({TIME_LEN, BATCH_SIZE, NUM_UNIT}));
  // cs_zero_tensor.tensor<float, 3>().setZero();
  // auto assign_cs = Assign(this->scope, cs, ZerosLike(this->scope, cs_zero_tensor));

  auto assign_ada_w = Assign(this->scope, ada_w, ZerosLike(this->scope, w));

  auto assign_ada_b = Assign(this->scope, ada_b, ZerosLike(this->scope, b));

  auto assign_ada_w_y = Assign(this->scope, ada_w_y, ZerosLike(this->scope, w_y));

  auto assign_ada_b_y = Assign(this->scope, ada_b_y, ZerosLike(this->scope, b_y));

  TF_CHECK_OK(session.Run({assign_w, assign_b, assign_w_y, assign_b_y},
                          nullptr));
  TF_CHECK_OK(session.Run({assign_ada_w, assign_ada_b, assign_ada_w_y, assign_ada_b_y},
                          nullptr));

  return Status::OK();
}

Status NoteRNN::Restore(const string graph_path) {
  Status load_graph_status = session.RestoreModel(graph_path);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
  }

  return load_graph_status;
}

Status NoteRNN::UpdateState(const Tensor &h, const Tensor &c) {
  CHECK(this->h_prev_tensor.CopyFrom(h.Slice(0, 1), {h.dim_size(1), h.dim_size(2)}));
  CHECK(this->cs_prev_tensor.CopyFrom(c.Slice(0, 1), {c.dim_size(1), c.dim_size(2)}));

  return Status::OK();
}

}  // namespace tensorflow
