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

// Wrap Variable, listed if trainable
class TFVariable {
 public:
  TFVariable(const ::tensorflow::Scope& scope, PartialTensorShape shape,
             DataType dtype, bool trainable = false);
  TFVariable(const ::tensorflow::Scope& scope, PartialTensorShape shape,
             DataType dtype, const Variable::Attrs& attrs,
             bool trainable = false);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }

  ::tensorflow::Output output;
};

// Wrap Assign, listed so it will be managed to be initialized
class TFAssign {
 public:
  TFAssign(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input value);
  TFAssign(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input value, const Assign::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }

  ::tensorflow::Output output;
};

class Moments {
 public:
  Moments(const ::tensorflow::Scope& scope, const ::tensorflow::Input& x,
          const std::initializer_list<int>& axes, bool keep_dims);

  ::tensorflow::Output mean;
  ::tensorflow::Output variance;
};

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

class KBatchNormalization {
 public:
  KBatchNormalization(const ::tensorflow::Scope& scope,
                      const ::tensorflow::Input& x,
                      const std::initializer_list<int>& axes,
                      PartialTensorShape shape,
                      const ::tensorflow::Input& variance_epsilon);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }

  ::tensorflow::Output moving_mean;
  ::tensorflow::Output moving_variance;

  ::tensorflow::Output assign_moving_mean;
  ::tensorflow::Output assign_moving_variance;
  ::tensorflow::Output update_moving_mean;
  ::tensorflow::Output update_moving_variance;

  ::tensorflow::Output gamma;
  ::tensorflow::Output gamma_m;
  ::tensorflow::Output gamma_v;
  ::tensorflow::Output beta;
  ::tensorflow::Output beta_m;
  ::tensorflow::Output beta_v;

  ::tensorflow::Output assign_gamma;
  ::tensorflow::Output assign_gamma_m;
  ::tensorflow::Output assign_gamma_v;
  ::tensorflow::Output assign_beta;
  ::tensorflow::Output assign_beta_m;
  ::tensorflow::Output assign_beta_v;

  ::tensorflow::Output output;
};

class Generator {
 public:
  explicit Generator(const ::tensorflow::Scope& scope);

  Output Build(const ::tensorflow::Scope& scope, const int batch_size);

 private:
  ::tensorflow::Output w1;
  ::tensorflow::Output filter;
  ::tensorflow::Output filter2;
  ::tensorflow::Output filter3;
};

class Discriminator {
 public:
  explicit Discriminator(const ::tensorflow::Scope& scope);

  Output Build(const ::tensorflow::Scope& scope,
               const ::tensorflow::Input& inputs, const int batch_size);

 private:
  ::tensorflow::Output conv1_weights;
  ::tensorflow::Output conv1_biases;
  ::tensorflow::Output conv2_weights;
  ::tensorflow::Output conv2_biases;
  ::tensorflow::Output fc1_weights;
  ::tensorflow::Output fc1_biases;
};

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_EXAMPLES_CC_GAN_DCGAN_NN_OPS_RKZ_H_
