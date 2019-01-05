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

#ifndef TENSORFLOW_CORE_KERNELS_RNN_OPS_H_
#define TENSORFLOW_CORE_KERNELS_RNN_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_activations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
class OpKernelContext;

namespace functor {

template <typename Device, typename T>
struct TensorZero {
  void operator()(const Device& d, typename TTypes<T>::Flat t) {
    t.device(d) = t.constant(T(0));
  }
};

template <typename Device, typename T>
struct TensorUnalignedZero {
  void operator()(const Device& d, typename TTypes<T>::UnalignedFlat t) {
    t.device(d) = t.constant(T(0));
  }
};

template <typename Device, typename T>
struct TensorCopy {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat src,
                  typename TTypes<T>::Flat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorCopyUnaligned {
  void operator()(const Device& d, typename TTypes<T>::UnalignedConstFlat src,
                  typename TTypes<T>::Flat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorCopyToUnaligned {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat src,
                  typename TTypes<T>::UnalignedFlat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorAdd {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat a,
                  typename TTypes<T>::ConstFlat b, typename TTypes<T>::Flat c) {
    c.device(d) = a + b;
  }
};

template <typename Device, typename T>
struct TensorZeroPadding {
  void operator()(const Device& d, const int64 time_idx,
                  typename TTypes<int64>::ConstVec seq_len,
                  typename TTypes<T>::Vec mask, typename TTypes<T>::Matrix m) {
    // mask is shape [batch_size].
    mask.device(d) = seq_len.constant(time_idx) < seq_len;

    // m_shape is [batch_size, 1].
    Eigen::array<Eigen::DenseIndex, 2> m_shape({m.dimensions()[0], 1});
    // broadcast_shape is [1, units].
    Eigen::array<Eigen::DenseIndex, 2> broadcast_shape({1, m.dimensions()[1]});

    // m is shape [batch_size, units].
    m.device(d) = m * mask.reshape(m_shape).broadcast(broadcast_shape);
  }
};

struct VanillaRNNCell {
  VanillaRNNCell(const int64 seq_length, const int64 input_size, const int64 hidden_size)
      : seq_length_(seq_length),
        input_size_(input_size),
        hidden_size_(hidden_size) {}

  int64 seq_length() const { return seq_length_; }
  int64 input_size() const { return input_size_; }
  int64 hidden_size() const { return hidden_size_; }

 protected:
  const int64 seq_length_;
  const int64 input_size_;
  const int64 hidden_size_;
};

// See rnn_ops.cc for CPUDevice implementation and rnn_ops_gpu.cu.cc (TODO) for
// GPUDevice implementation.
template <typename Device, typename T, bool USE_CUBLAS>
struct VanillaRNNCellFprop : public VanillaRNNCell {
  VanillaRNNCellFprop(const int64 seq_length, const int64 input_size, const int64 hidden_size)
      : VanillaRNNCell(seq_length, input_size, hidden_size) {}

  void operator()(OpKernelContext* ctx, const Device& d, const int64 t,
    typename TTypes<T>::ConstMatrix x,
    typename TTypes<T>::ConstScalar y,
    typename TTypes<T>::ConstMatrix h_prev,
    typename TTypes<T>::ConstMatrix w_xh,
    typename TTypes<T>::ConstMatrix w_hh,
    typename TTypes<T>::ConstMatrix w_hy,
    typename TTypes<T>::ConstMatrix b_h,
    typename TTypes<T>::ConstMatrix b_y,
    typename TTypes<T>::Matrix p_out, typename TTypes<T>::Matrix h_out,
    typename TTypes<T>::Scalar loss_out);
};

template <typename Device, typename T, bool USE_CUBLAS>
struct VanillaRNNCellBprop : public VanillaRNNCell {
  VanillaRNNCellBprop(const int64 seq_length, const int64 input_size, const int64 hidden_size)
      : VanillaRNNCell(seq_length, input_size, hidden_size) {}

  void operator()(
      OpKernelContext* ctx, const Device& d, const int64 t,                      
      typename TTypes<T>::ConstMatrix x,                                      
      typename TTypes<T>::ConstScalar y,                                      
      typename TTypes<T>::ConstMatrix p,                                 
      typename TTypes<T>::ConstMatrix h,                                 
      typename TTypes<T>::ConstMatrix w_hh,                                 
      typename TTypes<T>::ConstMatrix w_hy,
      typename TTypes<T>::ConstMatrix h_prev,
      typename TTypes<T>::Matrix dh_next,                                 
      typename TTypes<T>::Matrix d_w_xh_out,                                      
      typename TTypes<T>::Matrix d_w_hh_out,                                      
      typename TTypes<T>::Matrix d_w_hy_out,                                      
      typename TTypes<T>::Matrix d_b_h_out,                                       
      typename TTypes<T>::Matrix d_b_y_out);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RNN_OPS_H_
