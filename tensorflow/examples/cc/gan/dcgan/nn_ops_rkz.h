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
#ifndef TENSORFLOW_EXAMPLES_CC_GAN_DCGAN_NN_OPS_RKZ_H_
#define TENSORFLOW_EXAMPLES_CC_GAN_DCGAN_NN_OPS_RKZ_H_

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

class BatchNormalization {
 public:
  BatchNormalization(const ::tensorflow::Scope& scope,
                     const ::tensorflow::Input& x,
                     const ::tensorflow::Input& mean,
                     const ::tensorflow::Input& variance,
                     const ::tensorflow::Input& offset,
                     const ::tensorflow::Input& scale,
                     const ::tensorflow::Input& variance_epsilon);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }

  ::tensorflow::Output output;
};

class Dropout {
 public:
  Dropout(const ::tensorflow::Scope& scope, const ::tensorflow::Input x,
          const int rate);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }

  ::tensorflow::Output output;
};

class SigmoidCrossEntropyWithLogits {
 public:
  SigmoidCrossEntropyWithLogits(const ::tensorflow::Scope& scope,
                                const ::tensorflow::Input labels,
                                const ::tensorflow::Input logits);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }

  ::tensorflow::Output output;
};

class GlorotUniform {
 public:
  GlorotUniform(const ::tensorflow::Scope& scope,
                const std::initializer_list<int64>& shape);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }

  ::tensorflow::Output output;
};

class Conv2DTranspose {
 public:
  Conv2DTranspose(const ::tensorflow::Scope& scope,
                  const ::tensorflow::Input& input_sizes,
                  const ::tensorflow::Input& filter,
                  const ::tensorflow::Input& out_backprop,
                  const gtl::ArraySlice<int>& strides,
                  const StringPiece padding);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }

  ::tensorflow::Output output;
};

class Generator {
 public:
  Generator(const ::tensorflow::Scope& scope, const int batch_size);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }

  ::tensorflow::Output w1;
  ::tensorflow::Output w1_wm;
  ::tensorflow::Output w1_wv;
  ::tensorflow::Output filter;
  ::tensorflow::Output filter_wm;
  ::tensorflow::Output filter_wv;
  ::tensorflow::Output filter2;
  ::tensorflow::Output filter2_wm;
  ::tensorflow::Output filter2_wv;
  ::tensorflow::Output filter3;
  ::tensorflow::Output filter3_wm;
  ::tensorflow::Output filter3_wv;

  ::tensorflow::Output assign_w1;
  ::tensorflow::Output assign_filter;
  ::tensorflow::Output assign_filter2;
  ::tensorflow::Output assign_filter3;
  ::tensorflow::Output assign_w1_wm;
  ::tensorflow::Output assign_w1_wv;
  ::tensorflow::Output assign_filter_wm;
  ::tensorflow::Output assign_filter_wv;
  ::tensorflow::Output assign_filter2_wm;
  ::tensorflow::Output assign_filter2_wv;
  ::tensorflow::Output assign_filter3_wm;
  ::tensorflow::Output assign_filter3_wv;

  ::tensorflow::Output output;
};

class Discriminator {
 public:
  Discriminator(const ::tensorflow::Scope& scope,
                const ::tensorflow::Input& inputs, const int batch_size);
  Discriminator(const ::tensorflow::Scope& scope, const Discriminator& disc,
                const ::tensorflow::Input& inputs, const int batch_size);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }

  ::tensorflow::Output conv1_weights;
  ::tensorflow::Output conv1_wm;
  ::tensorflow::Output conv1_wv;
  ::tensorflow::Output conv1_biases;
  ::tensorflow::Output conv1_bm;
  ::tensorflow::Output conv1_bv;
  ::tensorflow::Output conv2_weights;
  ::tensorflow::Output conv2_wm;
  ::tensorflow::Output conv2_wv;
  ::tensorflow::Output conv2_biases;
  ::tensorflow::Output conv2_bm;
  ::tensorflow::Output conv2_bv;
  ::tensorflow::Output fc1_weights;
  ::tensorflow::Output fc1_wm;
  ::tensorflow::Output fc1_wv;
  ::tensorflow::Output fc1_biases;
  ::tensorflow::Output fc1_bm;
  ::tensorflow::Output fc1_bv;

  ::tensorflow::Output assign_conv1_weights;
  ::tensorflow::Output assign_conv1_biases;
  ::tensorflow::Output assign_conv2_weights;
  ::tensorflow::Output assign_conv2_biases;
  ::tensorflow::Output assign_fc1_weights;
  ::tensorflow::Output assign_fc1_biases;
  ::tensorflow::Output assign_conv1_wm;
  ::tensorflow::Output assign_conv1_wv;
  ::tensorflow::Output assign_conv1_bm;
  ::tensorflow::Output assign_conv1_bv;
  ::tensorflow::Output assign_conv2_wm;
  ::tensorflow::Output assign_conv2_wv;
  ::tensorflow::Output assign_conv2_bm;
  ::tensorflow::Output assign_conv2_bv;
  ::tensorflow::Output assign_fc1_wm;
  ::tensorflow::Output assign_fc1_wv;
  ::tensorflow::Output assign_fc1_bm;
  ::tensorflow::Output assign_fc1_bv;

  ::tensorflow::Output output;
};

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_EXAMPLES_CC_GAN_DCGAN_NN_OPS_RKZ_H_
