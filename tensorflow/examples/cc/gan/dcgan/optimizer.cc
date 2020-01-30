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
#include <algorithm>
#include <vector>

#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/examples/cc/gan/dcgan/nn_ops_rkz.h"
#include "tensorflow/examples/cc/gan/dcgan/optimizer.h"
#include "tensorflow/examples/cc/gan/dcgan/util.h"

namespace tensorflow {
namespace ops {

AdamOptimizer::AdamOptimizer(const ::tensorflow::Scope& scope) {
  this->lr = Const<float>(scope, LEARNING_RATE);
  this->beta1 = Const<float>(scope, BETA_1);
  this->beta2 = Const<float>(scope, BETA_2);
  this->epsilon = Const<float>(scope, EPSILON);

  this->global_step = Variable(scope, {}, DT_FLOAT);
  TFAssign(scope, global_step, 1.0f);

  this->assign_add_global_step = AssignAdd(scope, global_step, 1.0f);

  this->beta1_power = Pow(scope, beta1, global_step);
  this->beta2_power = Pow(scope, beta2, global_step);
}

void AdamOptimizer::Build(const ::tensorflow::Scope& scope,
                          std::vector<Output> outputs,
                          std::vector<Output> trainable_variables) {
  std::vector<Output> grad_outputs;
  TF_CHECK_OK(
      AddSymbolicGradients(scope, outputs, trainable_variables, &grad_outputs));
  LOG(INFO) << "Node building status: " << scope.status();

  int grad_index = 0;

  for (auto var : trainable_variables) {
    auto wm =
        Variable(scope, scope.GetTrainableVariableShape(var.node()->name()),
                 (DataType)(static_cast<int>(var.type()) - 100));
    LOG(INFO) << "Node building status: " << scope.status();
    TFAssign(scope, wm, ZerosLike(scope, var));
    auto wv =
        Variable(scope, scope.GetTrainableVariableShape(var.node()->name()),
                 (DataType)(static_cast<int>(var.type()) - 100));
    LOG(INFO) << "Node building status: " << scope.status();
    TFAssign(scope, wv, ZerosLike(scope, var));

    auto apply_adam =
        ApplyAdam(scope, var, wm, wv, beta1_power, beta2_power, lr, beta1,
                  beta2, epsilon, grad_outputs[grad_index++]);
    LOG(INFO) << "Node building status: " << scope.status();

    apply_adams.emplace_back(apply_adam);
  }

  for (auto var : outputs) {
    all_outputs.emplace_back(var);
  }
}

Status AdamOptimizer::Run(const ::tensorflow::Scope& scope,
                          const ::tensorflow::ClientSession& session,
                          std::vector<::tensorflow::Tensor>* outputs) {
  std::vector<Output> fetch_outputs;
  fetch_outputs.reserve(this->all_outputs.size() + this->apply_adams.size());
  fetch_outputs.insert(fetch_outputs.end(), this->all_outputs.begin(),
                       this->all_outputs.end());
  fetch_outputs.insert(fetch_outputs.end(), this->apply_adams.begin(),
                       this->apply_adams.end());

  Status status = session.Run(fetch_outputs, outputs);
  if (status.ok()) {
    std::vector<Tensor> assign_add_outputs;
    TF_CHECK_OK(
        session.Run({this->assign_add_global_step}, &assign_add_outputs));

    int step = static_cast<int>(assign_add_outputs[0].scalar<float>()());
    if (step % EVAL_FREQUENCY == 0) {
      LOG(INFO) << "Print step: " << step;
      for (int i = 0; i < this->all_outputs.size(); i++) {
        LOG(INFO) << "Print all_outputs " << i << ": "
                  << outputs->at(i).DebugString();
      }
    }
  }

  return status;
}

}  // namespace ops
}  // namespace tensorflow
