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
// rnn ops, Vanilla RNN for now
// Author: Rock Zhuang
// Date  : Dec 20, 2018
// 

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {

TEST(RNNOpsTest, VanillaRNN_InvalidNumberOfInputs) {
  ShapeInferenceTestOp op("VanillaRNN");
  op.input_tensors.resize(2);

  auto rebuild_node_def = [&op](const int hidsize) {
    TF_ASSERT_OK(NodeDefBuilder("test", "VanillaRNN")
                     .Input("x", 0, DT_FLOAT)
                     .Input("y", 0, DT_FLOAT)
                     .Input("h_pre", 0, DT_FLOAT)
                     .Input("w_xh", 0, DT_FLOAT)
                     .Input("w_hh", 0, DT_FLOAT)
                     .Input("w_hy", 0, DT_FLOAT)
                     .Input("b_h", 0, DT_FLOAT)
                     .Input("b_y", 0, DT_FLOAT)
                     .Attr("hidsize", hidsize)
                     .Finalize(&op.node_def));
  };

  rebuild_node_def(0);

  INFER_ERROR("Wrong number of inputs passed", op, "?;?");

  // INFER_OK(op, "[?,?];?;?", "[d0_0, d0_1];[d0_0, d0_1]");
}

// TEST(RNNOpsTest, DenseToDenseShape) {
//   ShapeInferenceTestOp op("VanillaRNN");

//   // Unknown shapes.
//   INFER_OK(op, "?;?", "[?,?];[?];[?]");

//   // Invalid rank.
//   INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[?];?");
//   INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "?;[?]");
//   INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[2];?");
//   INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "?;[2]");

//   // Mismatched ranks.
//   INFER_ERROR("Shape must be rank 2 but is rank 3", op, "[?,?];[?,?,?]");
//   INFER_ERROR("Shape must be rank 3 but is rank 2", op, "[?,?,?];[?,?]");
//   INFER_ERROR("Shape must be rank 2 but is rank 3", op, "[2,1];[2,1,2]");
//   INFER_ERROR("Shape must be rank 3 but is rank 2", op, "[2,1,2];[2,1]");

//   // Rank 2, unknown dims.
//   INFER_OK(op, "[?,?];?", "[?,2];[?];[2]");
//   INFER_OK(op, "?;[?,?]", "[?,2];[?];[2]");
//   INFER_OK(op, "[?,?];[?,?]", "[?,2];[?];[2]");

//   // Rank 4, unknown dims.
//   INFER_OK(op, "[?,?,?,?];?", "[?,4];[?];[4]");
//   INFER_OK(op, "?;[?,?,?,?]", "[?,4];[?];[4]");
//   INFER_OK(op, "[?,?,?,?];[?,?,?,?]", "[?,4];[?];[4]");

//   // Known rank for 1 input.
//   INFER_OK(op, "[5,3,2,1];?", "[?,4];[?];[4]");
//   INFER_OK(op, "?;[5,3,2,1]", "[?,4];[?];[4]");
//   INFER_OK(op, "[5,3,2,1];[?,?,?,?]", "[?,4];[?];[4]");
//   INFER_OK(op, "[?,?,?,?];[5,3,2,1]", "[?,4];[?];[4]");
//   INFER_OK(op, "[5,3,2,1];[?,?,?,?]", "[?,4];[?];[4]");

//   // Mismatched n-1 dims.
//   INFER_ERROR("Dimension 0 in both shapes must be equal", op,
//               "[4,?,2,?];[3,1,?,5]");
//   INFER_ERROR("Dimension 2 in both shapes must be equal", op,
//               "[4,3,2,1];[4,3,3,1]");

//   // Matched n-1 dims.
//   INFER_OK(op, "[4,5,6,7];[?,?,?,?]", "[?,4];[?];[4]");
//   INFER_OK(op, "[4,5,6,7];[?,?,?,4]", "[?,4];[?];[4]");
//   INFER_OK(op, "[?,?,?,?];[4,5,6,7]", "[?,4];[?];[4]");
//   INFER_OK(op, "[4,?,2,?];[?,1,?,5]", "[?,4];[?];[4]");
//   INFER_OK(op, "[4,5,6,7];[4,?,6,?]", "[?,4];[?];[4]");
//   INFER_OK(op, "[4,5,6,7];[4,5,6,4]", "[?,4];[?];[4]");
// }

TEST(RNNOpsTest, VanillaRNNGrad_InvalidNumberOfInputs) {
  ShapeInferenceTestOp op("VanillaRNNGrad");
  op.input_tensors.resize(2);

  auto rebuild_node_def = [&op](const int hidsize) {
    TF_ASSERT_OK(NodeDefBuilder("test", "VanillaRNNGrad")
                     .Input("x", 0, DT_FLOAT)
                     .Input("y", 0, DT_FLOAT)
                     .Input("p", 0, DT_FLOAT)
                     .Input("h", 0, DT_FLOAT)
                     .Input("w_hh", 0, DT_FLOAT)
                     .Input("w_hy", 0, DT_FLOAT)
                     .Input("h_prev", 0, DT_FLOAT)
                     .Attr("hidsize", hidsize)
                     .Finalize(&op.node_def));
  };

  // Default squeeze_dims = []
  rebuild_node_def(10);

  INFER_ERROR("Wrong number of inputs passed", op, "?;?;?");
  INFER_ERROR("Shape must be rank 3 but is rank 2", op, "?;?;[?,?];?;?;?;?");
  INFER_ERROR("Shape must be rank 3 but is rank 2", op, "?;?;[?,?,?];[?,?];?;?;?");
}

}  // end namespace tensorflow
