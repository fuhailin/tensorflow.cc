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


//
// RKZConv2D and RKZConv2DGrad are examples of convolution layer
// Ref: http://cs231n.github.io/convolutional-networks/
// Ref: https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
//
// Author: Rock Zhuang
// Date  : Feb 07, 2019
//

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#if GOOGLE_CUDA_TODO
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA_TODO

#include "tensorflow/core/kernels/rkz_conv_ops.h"

#include <string.h>
#include <map>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/tensor_format.h"

// #define VERBOSE 1
// #define VERBOSE_FP 1
// #define VERBOSE_BPF 1
// #define VERBOSE_BPI 1
// #define VERBOSE_IM2COL 1
// #define VERBOSE_COL2IM 1
// #define TESTING 1

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

static string DetailedDebugString(const Tensor &tensor) {
  return strings::StrCat("Tensor<type: ", DataTypeString(tensor.dtype()),
                         " shape: ", tensor.shape().DebugString(),
                         " values: ", tensor.SummarizeValue(-1, true), ">");
}

// Pad input with padding size of 'padding'
// shape of in is {batch, height, width, channels }
// shape of out is {batch, out_height, out_width, channels }
template <typename T>
struct PadInput<CPUDevice, T> {
  void operator()(typename TTypes<T, 4>::ConstTensor in, 
              int padding, 
              typename TTypes<T, 4>::Tensor out) {

    Eigen::array<std::pair<int, int>, 4> paddings;
    paddings[0] = std::make_pair(0, 0);
    paddings[1] = std::make_pair(padding, padding); // H
    paddings[2] = std::make_pair(padding, padding); // W
    paddings[3] = std::make_pair(0, 0);

    out = in.pad(paddings);
  }
};

template <typename T>
struct UnpadOutput<CPUDevice, T> {
  void operator()(typename TTypes<T, 4>::ConstTensor in, 
              int padding, 
              typename TTypes<T, 4>::Tensor out) {

    Eigen::array<std::pair<int, int>, 4> paddings;
    paddings[0] = std::make_pair(0, 0);
    paddings[1] = std::make_pair(-padding, -padding); // H
    paddings[2] = std::make_pair(-padding, -padding); // W
    paddings[3] = std::make_pair(0, 0);

    out = in.pad(paddings);
  }
};

// for one image with multiple channels
// Input is batch size of images with shape {batch, height, width, channels}
// shape of out: {}
template <typename T>
struct Im2Col<CPUDevice, T> {
  void operator()(typename TTypes<T, 4>::ConstTensor in, 
            const TensorShape& filter_shape,
            int out_h, int out_w,
            typename TTypes<T>::Matrix out) {
    int im2col_out_d0 = filter_shape.dim_size(0) * filter_shape.dim_size(1) * filter_shape.dim_size(2);

    int batch_size = in.dimension(0);
    // int channels = in.dimension(3);

    for(int bidx = 0; bidx < batch_size; bidx++) {
      // Chip 
      Eigen::Tensor<T, 3, Eigen::RowMajor> image_et = in.chip(bidx, 0);
#ifdef VERBOSE_IM2COL
      // LOG(INFO) << "----------------Im2Col image_et rank: " << image_et.rank();
      LOG(INFO) << "----------------Im2Col image_et: " << image_et;
#endif

      // 
      for(int hidx = 0; hidx < out_h; hidx++) {
        for(int widx = 0; widx < out_w; widx++) {

          Eigen::array<int, 3> offsets = {hidx, widx, 0};
          Eigen::array<int, 3> extents = {filter_shape.dim_size(0), filter_shape.dim_size(1), filter_shape.dim_size(2)};
          Eigen::Tensor<T, 3, Eigen::RowMajor> image_et_slice = image_et.slice(offsets, extents);
#ifdef VERBOSE_IM2COL
      // LOG(INFO) << "----------------Im2Col image_et_slice rank: " << image_et_slice.rank();
      LOG(INFO) << "----------------Im2Col image_et_slice: " << image_et_slice;
#endif
          
          Eigen::array<int, 1> one_dim{{im2col_out_d0}};
          out.chip(widx + hidx * out_w + bidx * out_h * out_w, 1) = image_et_slice.reshape(one_dim);
#ifdef VERBOSE_IM2COL
      // LOG(INFO) << "----------------Im2Col image_et_slice.reshape(one_dim): " << image_et_slice.reshape(one_dim);
#endif
        }
      }
    }
  
  }
};

template <typename T>
struct Col2Im<CPUDevice, T> {
  void operator()(typename TTypes<T>::ConstMatrix in, 
            const TensorShape& input_shape,
            const TensorShape& filter_shape,
            int out_h, int out_w,
            typename TTypes<T, 4>::Tensor out) {

#ifdef VERBOSE_COL2IM
      LOG(INFO) << "----------------Col2Im in: " << std::endl << in;
#endif

    int batch_size = input_shape.dim_size(0);

    for(int bidx = 0; bidx < batch_size; bidx++) {
      // one image from batches
      Eigen::Tensor<T, 3, Eigen::RowMajor> image_et(input_shape.dim_size(1), input_shape.dim_size(2), input_shape.dim_size(3));

      // 
      for(int hidx = 0; hidx < out_h; hidx++) {
        for(int widx = 0; widx < out_w; widx++) {
          // reshape 'in' from vec to three dims
          Eigen::array<int, 3> three_dims{{filter_shape.dim_size(0), filter_shape.dim_size(1), filter_shape.dim_size(2)}};

          Eigen::array<int, 3> offsets = {hidx, widx, 0};
          Eigen::array<int, 3> extents = {filter_shape.dim_size(0), filter_shape.dim_size(1), filter_shape.dim_size(2)};

          image_et.slice(offsets, extents) = in.chip(widx + hidx * out_w + bidx * out_h * out_w, 1).reshape(three_dims);
        }
      }

      out.chip(bidx, 0) = image_et;
#ifdef VERBOSE_COL2IM
      LOG(INFO) << "----------------Col2Im image_et: " << image_et;
#endif      
    }
    
  }
};

template <typename T>
struct ComputeResult<CPUDevice, T> {
  void operator()(typename TTypes<T>::ConstMatrix im2col_out, 
            typename TTypes<T, 4>::ConstTensor filter, 
            int batch_size, int out_h, int out_w,
            typename TTypes<T, 4>::Tensor out) {
    
    // reshape filter
    int filter_reshaped_d0 = filter.dimension(0) * filter.dimension(1) * filter.dimension(2);
    Eigen::array<int, 2> two_dims{{filter_reshaped_d0, filter.dimension(3)}};
    auto W_row = filter.reshape(two_dims);

    // Mat Mul
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(0, 0) }; // for dot product
    auto result_two_dims = im2col_out.contract(W_row, product_dims);
    // Eigen::array<Eigen::DenseIndex, 2> b_shape({1, filter.dimension(3)});
    // Eigen::array<Eigen::DenseIndex, 2> b_cast({im2col_out.dimension(1), 1});
    // auto result_two_dims = im2col_out.contract(W_row, product_dims) + bias.reshape(b_shape).broadcast(b_cast);

    Eigen::array<int, 4> four_dims{{batch_size, out_h, out_w, filter.dimension(3)}};
    out = result_two_dims.reshape(four_dims);  
  }
};

template <typename T>
void test(OpKernelContext* ctx) {
    int stride = 1;
    int padding = 1;

    int index = 1;

    Tensor input(DT_FLOAT, TensorShape({2, 5, 5, 3}));
    for (int64 a = 0; a < input.shape().dim_size(0); a++) {
      for (int64 b = 0; b < input.shape().dim_size(1); b++) {
        for (int64 c = 0; c < input.shape().dim_size(2); c++) {
          for (int64 d = 0; d < input.shape().dim_size(3); d++) {
            input.tensor<float, 4>()(a, b, c, d) = index++;
          }
        }
      }
    }

    LOG(INFO) << "----------------test input: " << DetailedDebugString(input);

    index = 1;
    Tensor filter(DT_FLOAT, TensorShape({2, 2, 3, 5}));
    for (int64 a = 0; a < filter.shape().dim_size(0); a++) {
      for (int64 b = 0; b < filter.shape().dim_size(1); b++) {
        for (int64 c = 0; c < filter.shape().dim_size(2); c++) {
          for (int64 d = 0; d < filter.shape().dim_size(3); d++) {
            filter.tensor<float, 4>()(a, b, c, d) = index++;
          }
        }
      }
    }

    LOG(INFO) << "----------------test filter: " << DetailedDebugString(filter);

    // index = 1;
    // Tensor bias(DT_FLOAT, TensorShape({filter.dim_size(3)}));
    // for (int64 a = 0; a < bias.shape().dim_size(0); a++) {
    //   bias.vec<float>()(a) = index++;
    // }
    // LOG(INFO) << "----------------test bias: " << DetailedDebugString(bias);

    int batch_size = input.dim_size(0);

#if 0
    // Im2Col without padding
    int out_h = (input.dim_size(1) - filter.dim_size(0)) / stride + 1;
    int out_w = (input.dim_size(2) - filter.dim_size(1)) / stride + 1;

    int im2col_out_d0 = filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);
    int im2col_out_d1 = out_h * out_w * batch_size;

    Tensor im2col_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({im2col_out_d0, im2col_out_d1}), &im2col_out));


    Im2Col<CPUDevice, T> im2col;
    im2col(((const Tensor)input).tensor<T, 4>(), filter.shape(), out_h, out_w, im2col_out.matrix<T>());

    LOG(INFO) << "----------------test im2col_out: " << DetailedDebugString(im2col_out);

    // Col2Im
    Tensor col2im_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           input.shape(), &col2im_out));

    Col2Im<CPUDevice, T> col2im;
    col2im(((const Tensor)im2col_out).matrix<T>(), input.shape(), filter.shape(), out_h, out_w, col2im_out.tensor<T, 4>());

    LOG(INFO) << "----------------test col2im_out: " << DetailedDebugString(col2im_out);
#else
    // Pad
    int channels = input.dim_size(3);

    int h2 = input.dim_size(1) + 2 * padding;
    int w2 = input.dim_size(2) + 2 * padding; 

    Tensor pad_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, h2, w2, channels}), &pad_out));

    PadInput<CPUDevice, T> pad_input;
    pad_input(((const Tensor)input).tensor<T, 4>(), padding, pad_out.tensor<T, 4>());

    LOG(INFO) << "----------------test pad_out: " << DetailedDebugString(pad_out);

    // Im2Col with padding
    int out_h = (input.dim_size(1) - filter.dim_size(0) + 2 * padding) / stride + 1;
    int out_w = (input.dim_size(2) - filter.dim_size(1) + 2 * padding) / stride + 1; 

    int im2col_out_d0 = filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);
    int im2col_out_d1 = out_h * out_w * batch_size;

    Tensor im2col_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({im2col_out_d0, im2col_out_d1}), &im2col_out));


    Im2Col<CPUDevice, T> im2col;
    im2col(((const Tensor)pad_out).tensor<T, 4>(), filter.shape(), out_h, out_w, im2col_out.matrix<T>());

    LOG(INFO) << "----------------test im2col_out: " << DetailedDebugString(im2col_out);

    // Col2Im
    Tensor col2im_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           pad_out.shape(), &col2im_out));

    Col2Im<CPUDevice, T> col2im;
    col2im(((const Tensor)im2col_out).matrix<T>(), pad_out.shape(), filter.shape(), out_h, out_w, col2im_out.tensor<T, 4>());

    LOG(INFO) << "----------------test col2im_out: " << DetailedDebugString(col2im_out);

    // Unpad 
    Tensor unpad_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           input.shape(), &unpad_out));

    UnpadOutput<CPUDevice, T> unpad_output;
    unpad_output(((const Tensor)col2im_out).tensor<T, 4>(), padding, unpad_out.tensor<T, 4>());

    LOG(INFO) << "----------------test unpad_out: " << DetailedDebugString(unpad_out);
#endif

    // ComputeResult
    Tensor result;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, out_h, out_w, filter.dim_size(3)}), &result));


    ComputeResult<CPUDevice, T> compute_result;
    compute_result(((const Tensor)im2col_out).matrix<T>(), ((const Tensor)filter).tensor<T, 4>(),
                   batch_size, out_h, out_w, 
                   result.tensor<T, 4>());

    LOG(INFO) << "----------------test result: " << DetailedDebugString(result);


}

template <typename T>
struct LaunchRKZConv2DOp<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, 
                  const Tensor& input, const Tensor& filter,
                  int stride, int padding,
                  Tensor* output) {

#ifdef TESTING
    test<T>(ctx);
#else
    // Pad
    int batch_size = input.dim_size(0);
    int channels = input.dim_size(3);

    int h2 = input.dim_size(1) + 2 * padding;
    int w2 = input.dim_size(2) + 2 * padding; 

    Tensor pad_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, h2, w2, channels}), &pad_out));


    PadInput<CPUDevice, T> pad_input;
    pad_input(input.tensor<T, 4>(), padding, pad_out.tensor<T, 4>());
#ifdef VERBOSE_FP
    LOG(INFO) << "----------------LaunchRKZConv2DOp pad_out: " << pad_out.DebugString();
#endif

    // Im2Col with padding
    int out_h = (input.dim_size(1) - filter.dim_size(0) + 2 * padding) / stride + 1;
    int out_w = (input.dim_size(2) - filter.dim_size(1) + 2 * padding) / stride + 1; 

    int im2col_out_d0 = filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);
    int im2col_out_d1 = out_h * out_w * batch_size;

    Tensor im2col_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({im2col_out_d0, im2col_out_d1}), &im2col_out));


    Im2Col<CPUDevice, T> im2col;
    im2col(((const Tensor)pad_out).tensor<T, 4>(), filter.shape(), out_h, out_w, im2col_out.matrix<T>());
#ifdef VERBOSE_FP
    LOG(INFO) << "----------------LaunchRKZConv2DOp im2col_out: " << im2col_out.DebugString();
#endif
    
    // ComputeResult
    ComputeResult<CPUDevice, T> compute_result;
    compute_result(((const Tensor)im2col_out).matrix<T>(), ((const Tensor)filter).tensor<T, 4>(),
                   batch_size, out_h, out_w, 
                   output->tensor<T, 4>());    
#endif

  }
};

template <typename Device, typename T>
class RKZConv2DOp : public BinaryOp<T> {
 public:
  explicit RKZConv2DOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("stride", &stride));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    // (W - F + 2P) / S + 1
    int out_rows = (input.dim_size(1) - filter.dim_size(0) + 2 * padding) / stride + 1;
    int out_cols = (input.dim_size(2) - filter.dim_size(1) + 2 * padding) / stride + 1; 

    TensorShape out_shape = ShapeFromFormat(
        FORMAT_NHWC, input.dim_size(0), out_rows,
        out_cols, filter.dim_size(3));

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    launcher_(context, input, filter,
              stride, padding, output);
  }

 private:
  int stride;
  int padding;

  LaunchRKZConv2DOp<Device, T> launcher_;

  TF_DISALLOW_COPY_AND_ASSIGN(RKZConv2DOp);
};

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("RKZConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RKZConv2DOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU


//
// Grad
//

// Filter backprop

template <typename T>
struct GradFilterComputeResult<CPUDevice, T> {
  void operator()(typename TTypes<T>::ConstMatrix im2col_out, 
            typename TTypes<T, 4>::ConstTensor dout,
            const TensorShape& filter_shape,
            typename TTypes<T, 4>::Tensor out) {
    
    // reshape filter
    int dout_reshaped_d0 = dout.dimension(0) * dout.dimension(1) * dout.dimension(2);
    Eigen::array<int, 2> two_dims{{dout_reshaped_d0, dout.dimension(3)}};
    auto dout_reshaped = dout.reshape(two_dims);

    // Mat Mul
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) }; // for dot product
    auto result_two_dims = im2col_out.contract(dout_reshaped, product_dims);

    Eigen::array<int, 4> four_dims{{filter_shape.dim_size(0), filter_shape.dim_size(1), filter_shape.dim_size(2), filter_shape.dim_size(3)}};
    out = result_two_dims.reshape(four_dims);  
  }
};

template <typename T>
struct LaunchRKZConv2DGradFilterOp<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, 
                  const Tensor& input, 
                  const TensorShape& filter_shape,
                  const Tensor& dout, 
                  int stride, int padding,
                  Tensor* output) {

    // Pad
    int batch_size = input.dim_size(0);
    int channels = input.dim_size(3);

    int h2 = input.dim_size(1) + 2 * padding;
    int w2 = input.dim_size(2) + 2 * padding; 

    Tensor pad_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, h2, w2, channels}), &pad_out));


    PadInput<CPUDevice, T> pad_input;
    pad_input(input.tensor<T, 4>(), padding, pad_out.tensor<T, 4>());
#ifdef VERBOSE_BPF
    LOG(INFO) << "----------------LaunchRKZConv2DGradFilterOp pad_out: " << pad_out.DebugString();
#endif

    // Im2Col with padding
    int out_h = (input.dim_size(1) - filter_shape.dim_size(0) + 2 * padding) / stride + 1;
    int out_w = (input.dim_size(2) - filter_shape.dim_size(1) + 2 * padding) / stride + 1; 

    int im2col_out_d0 = filter_shape.dim_size(0) * filter_shape.dim_size(1) * filter_shape.dim_size(2);
    int im2col_out_d1 = out_h * out_w * batch_size;

    Tensor im2col_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({im2col_out_d0, im2col_out_d1}), &im2col_out));


    Im2Col<CPUDevice, T> im2col;
    im2col(((const Tensor)pad_out).tensor<T, 4>(), filter_shape, out_h, out_w, im2col_out.matrix<T>());
#ifdef VERBOSE_BPF
    LOG(INFO) << "----------------LaunchRKZConv2DGradFilterOp im2col_out: " << im2col_out.DebugString();
#endif
    
    // ComputeResult
    GradFilterComputeResult<CPUDevice, T> compute_result;
    compute_result(((const Tensor)im2col_out).matrix<T>(), ((const Tensor)dout).tensor<T, 4>(),
                   filter_shape, 
                   output->tensor<T, 4>());

    
  }
};

template <typename Device, typename T>
class RKZConv2DGradFilterOp : public OpKernel {
 public:
  explicit RKZConv2DGradFilterOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("stride", &stride));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter size is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter_dims = context->input(1);

    // Input dout is of the following dimensions:
    // [ batch, dout_cols, dout_rows, n_filters]
    const Tensor& dout = context->input(2);

#ifdef VERBOSE_BPF
    LOG(INFO) << "----------------RKZConv2DGradFilterOp input: " << input.DebugString();
    LOG(INFO) << "----------------RKZConv2DGradFilterOp filter_dims: " << filter_dims.DebugString();
    LOG(INFO) << "----------------RKZConv2DGradFilterOp dout: " << dout.DebugString();
#endif

    TensorShape filter_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                filter_dims.vec<int32>(), &filter_shape));

    // Output tensor is of the same shape as filter
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, filter_shape, &output));

    launcher_(context, input, filter_shape, dout,
              stride, padding, output);
  }

 private:
  int stride;
  int padding;

  LaunchRKZConv2DGradFilterOp<Device, T> launcher_;

  TF_DISALLOW_COPY_AND_ASSIGN(RKZConv2DGradFilterOp);
};

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("RKZConv2DGradFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RKZConv2DGradFilterOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU

// Input backprop

template <typename T>
struct GradInputComputeResult<CPUDevice, T> {
  void operator()(typename TTypes<T, 4>::ConstTensor filter, 
            typename TTypes<T, 4>::ConstTensor dout,
            typename TTypes<T>::Matrix out) {
    
    // reshape filter
    int filter_reshaped_d0 = filter.dimension(0) * filter.dimension(1) * filter.dimension(2);
    Eigen::array<int, 2> two_dims_filter{{filter_reshaped_d0, filter.dimension(3)}};
    auto filter_reshaped = filter.reshape(two_dims_filter);

    // reshape dout
    int dout_reshaped_d0 = dout.dimension(0) * dout.dimension(1) * dout.dimension(2);
    Eigen::array<int, 2> two_dims{{dout_reshaped_d0, dout.dimension(3)}};
    auto dout_reshaped = dout.reshape(two_dims);

    // Mat Mul
    // Got shape: {filter_reshaped_d0, dout_reshaped_d0}
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 1) }; // for dot product
    out = filter_reshaped.contract(dout_reshaped, product_dims);

  }
};

template <typename T>
struct LaunchRKZConv2DGradInputOp<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, 
                  const TensorShape& input_shape, 
                  const Tensor& filter,
                  const Tensor& dout, 
                  int stride, int padding,
                  Tensor* output) {

    //
    // dX_col = W_reshape.T @ dout_reshaped
    //
    int filter_reshaped_d0 = filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);
    int dout_reshaped_d0 = dout.dim_size(0) * dout.dim_size(1) * dout.dim_size(2);
    Tensor dx_col_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({filter_reshaped_d0, dout_reshaped_d0}), &dx_col_out));

    // ComputeResult
    GradInputComputeResult<CPUDevice, T> compute_result;
    compute_result(((const Tensor)filter).tensor<T, 4>(), ((const Tensor)dout).tensor<T, 4>(),
                   dx_col_out.matrix<T>());

#ifdef VERBOSE_BPI
    LOG(INFO) << "----------------LaunchRKZConv2DGradInputOp dx_col_out: " << dx_col_out.DebugString();
#endif

    //
    // Col2im
    //

    // (W - F + 2P) / S + 1
    int out_h = (input_shape.dim_size(1) - filter.dim_size(0) + 2 * padding) / stride + 1;
    int out_w = (input_shape.dim_size(2) - filter.dim_size(1) + 2 * padding) / stride + 1; 


    TensorShape padout_shape({input_shape.dim_size(0), 
                                              input_shape.dim_size(1) + 2 * padding, 
                                              input_shape.dim_size(2) + 2 * padding, 
                                              input_shape.dim_size(3)});

    Tensor col2im_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           padout_shape, 
                                           &col2im_out));

    Col2Im<CPUDevice, T> col2im;
    col2im(((const Tensor)dx_col_out).matrix<T>(), padout_shape, filter.shape(), out_h, out_w, col2im_out.tensor<T, 4>());
#ifdef VERBOSE_BPI
    LOG(INFO) << "----------------LaunchRKZConv2DGradInputOp col2im_out: " << col2im_out.DebugString();
#endif

    // Unpad 
    UnpadOutput<CPUDevice, T> unpad_output;
    unpad_output(((const Tensor)col2im_out).tensor<T, 4>(), padding, output->tensor<T, 4>());
#ifdef VERBOSE_BPI
    LOG(INFO) << "----------------LaunchRKZConv2DGradInputOp output: " << output->DebugString();
#endif
  }
};

template <typename Device, typename T>
class RKZConv2DGradInputOp : public OpKernel {
 public:
  explicit RKZConv2DGradInputOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("stride", &stride));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input_dims = context->input(0);

    // Input filter size is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    // Input dout is of the following dimensions:
    // [ batch, dout_cols, dout_rows, n_filters]
    const Tensor& dout = context->input(2);

#ifdef VERBOSE_BPI
    LOG(INFO) << "----------------RKZConv2DGradInputOp input_dims: " << DetailedDebugString(input_dims);
    LOG(INFO) << "----------------RKZConv2DGradInputOp filter: " << filter.DebugString();
    LOG(INFO) << "----------------RKZConv2DGradInputOp dout: " << dout.DebugString();
#endif

    TensorShape input_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_dims.vec<int32>(), &input_shape));

    // Output tensor is of the same shape as input
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));

    launcher_(context, input_shape, filter, dout,
              stride, padding, output);
  }

 private:
  int stride;
  int padding;

  LaunchRKZConv2DGradInputOp<Device, T> launcher_;

  TF_DISALLOW_COPY_AND_ASSIGN(RKZConv2DGradInputOp);
};

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("RKZConv2DGradInput").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RKZConv2DGradInputOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU


#if GOOGLE_CUDA_TODO
// Registration of the GPU implementations.
REGISTER_KERNEL_BUILDER(
    Name("RKZConv2D").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    RKZConv2DOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("RKZConv2D").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    RKZConv2DOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("RKZConv2D").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    RKZConv2DOp<GPUDevice, double>);

// To be used inside depthwise_conv_op.cc.
template struct LaunchRKZConv2DOp<GPUDevice, float>;
template struct LaunchRKZConv2DOp<GPUDevice, Eigen::half>;
template struct LaunchRKZConv2DOp<GPUDevice, double>;

#endif  // GOOGLE_CUDA_TODO

}  // namespace tensorflow
