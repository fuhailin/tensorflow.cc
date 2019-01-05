/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//
// rnn ops, Vanilla RNN for now
// Author: Rock Zhuang
// Date  : Dec 20, 2018
// 

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("VanillaRNN")
    .Input("x: T")
    .Input("y: T")
    .Input("h_prev: T")
    .Input("w_xh: T")
    .Input("w_hh: T")
    .Input("w_hy: T")
    .Input("b_h: T")
    .Input("b_y: T")
    .Output("p: T")
    .Output("h: T")
    .Output("loss: T")
    .Attr("hidsize: int = 100")
    .Attr("T: {float}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, y, h_prev, w_xh, w_hh, w_hy, b_h, b_y;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &y));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &h_prev));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &w_xh));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &w_hh));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &w_hy));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &b_h));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 2, &b_y));

      DimensionHandle seq_length = c->Dim(x, 0);
      DimensionHandle input_size = c->Dim(x, 1);

      ShapeHandle output = c->MakeShape({seq_length, input_size, 1});
      c->set_output(0, output);

      int32 hidsize;
      TF_RETURN_IF_ERROR(c->GetAttr("hidsize", &hidsize));
      output = c->MakeShape({seq_length, hidsize, 1});
      c->set_output(1, output);

      output = c->Scalar();
      c->set_output(2, output);
      
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Computes the Vanilla RNN forward propagation for all time step.
)doc");


REGISTER_OP("VanillaRNNGrad")
    .Input("x: T")
    .Input("y: T")
    .Input("p: T")
    .Input("h: T")
    .Input("w_hh: T")
    .Input("w_hy: T")
    .Input("h_prev: T")
    .Output("d_w_xh: T")
    .Output("d_w_hh: T")
    .Output("d_w_hy: T")
    .Output("d_b_h: T")
    .Output("d_b_y: T")
    .Attr("hidsize: int = 100")
    .Attr("T: {float}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, y, p, h, w_hh, w_hy, h_prev;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &y));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &p));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &h));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &w_hh));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &w_hy));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &h_prev));
      
      DimensionHandle seq_length = c->Dim(x, 0);
      DimensionHandle input_size = c->Dim(x, 1);
      int32 hidsize;
      TF_RETURN_IF_ERROR(c->GetAttr("hidsize", &hidsize));

      ShapeHandle output = c->Matrix(hidsize, input_size);
      c->set_output(0, output);

      output = c->Matrix(hidsize, hidsize);
      c->set_output(1, output);

      output = c->Matrix(input_size, hidsize);
      c->set_output(2, output);

      output = c->Matrix(hidsize, 1);
      c->set_output(3, output);

      output = c->Matrix(input_size, 1);
      c->set_output(4, output);

      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Computes the Vanilla RNN back propagation for all time step.
)doc");

}  // end namespace tensorflow
