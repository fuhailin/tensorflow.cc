/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_RKZ_CONV_OPS_H_
#define TENSORFLOW_CORE_KERNELS_RKZ_CONV_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

// Forward declaration.
class OpKernelContext;

//
// Forward Prop
//

template <typename Device, typename T>
struct LaunchRKZConv2DOp {
  void operator()(OpKernelContext* ctx,
                  const Tensor& input, const Tensor& filter,
                  int stride, int padding,
                  Tensor* output);
};

template <typename Device, typename T>
struct PadInput {
  void operator()(typename TTypes<T, 4>::ConstTensor in, 
              int padding, 
              typename TTypes<T, 4>::Tensor out);
};

template <typename Device, typename T>
struct UnpadOutput {
  void operator()(typename TTypes<T, 4>::ConstTensor in, 
              int padding, 
              typename TTypes<T, 4>::Tensor out);
};

template <typename Device, typename T>
struct Im2Col {
  void operator()(typename TTypes<T, 4>::ConstTensor in, 
            const TensorShape& filter_shape,
            int out_h, int out_w,
            typename TTypes<T>::Matrix out);
};

template <typename Device, typename T>
struct Col2Im {
  void operator()(typename TTypes<T>::ConstMatrix in, 
            const TensorShape& input_shape,
            const TensorShape& filter_shape,
            int out_h, int out_w,
            typename TTypes<T, 4>::Tensor out);
};

template <typename Device, typename T>
struct ComputeResult {
  void operator()(typename TTypes<T>::ConstMatrix im2col_out, 
            typename TTypes<T, 4>::ConstTensor filter, 
            int batch_size, int out_h, int out_w,
            typename TTypes<T, 4>::Tensor out);
};

//
// Grad
//

// Filter
template <typename Device, typename T>
struct GradFilterComputeResult {
  void operator()(typename TTypes<T>::ConstMatrix im2col_out, 
            typename TTypes<T, 4>::ConstTensor dout, 
            const TensorShape& filter_shape,
            typename TTypes<T, 4>::Tensor out);
};

template <typename Device, typename T>
struct LaunchRKZConv2DGradFilterOp {
  void operator()(OpKernelContext* ctx, 
                  const Tensor& input, 
                  const TensorShape& filter_shape,
                  const Tensor& dout, 
                  int stride, int padding,
                  Tensor* output);
};

// Input
template <typename Device, typename T>
struct GradInputComputeResult {
  void operator()(typename TTypes<T, 4>::ConstTensor filter, 
            typename TTypes<T, 4>::ConstTensor dout,
            typename TTypes<T>::Matrix out);
};

template <typename Device, typename T>
struct LaunchRKZConv2DGradInputOp {
  void operator()(OpKernelContext* ctx, 
                  const TensorShape& input_shape, 
                  const Tensor& filter,
                  const Tensor& dout, 
                  int stride, int padding,
                  Tensor* output);
};


}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_H_
