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
// Bridge of RNN forward and backward propagation, 
// return loss, h_grad, dw_y, db_y values
// Ref: sparse_xent_op and lstm_ops modules
//
// Author: Rock Zhuang
// Date  : Jan 15, 2019
// 

#ifndef TENSORFLOW_CORE_KERNELS_RNN_SOFTMAXLOSS_HGRAD_OP_H_
#define TENSORFLOW_CORE_KERNELS_RNN_SOFTMAXLOSS_HGRAD_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/blas_gemm.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/default/logging.h"

namespace tensorflow {

namespace sparse_xent_helpers {

template <typename T>
typename TTypes<const T, 1>::Tensor32Bit To32BitConst(
    typename TTypes<T>::Vec in) {
  return To32Bit(typename TTypes<T>::ConstVec(in.data(), in.dimensions()));
}

template <typename T>
typename TTypes<const T, 2>::Tensor32Bit To32BitConst(
    typename TTypes<T>::Matrix in) {
  return To32Bit(typename TTypes<T>::ConstMatrix(in.data(), in.dimensions()));
}

}  // namespace sparse_xent_helpers

namespace generator {

// Generator for calculation of the sparse Xent loss.
// This generator takes the logits, the sum of the exponentiated
// logits, and the label indices.  For each minibatch entry, ignoring
// the batch index b, it calculates:
//
//   loss[j] = (log(sum_exp_logits) - logits[j]) * 1{ j == label }
//
// for j = 0 .. num_classes.  This value must be summed over all j for
// the final loss.
template <typename T, typename Index>
class SparseXentLossGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE SparseXentLossGenerator(
      typename TTypes<const T, 2>::Tensor32Bit logits,
      typename TTypes<const T, 1>::Tensor32Bit sum_exp_logits,
      typename TTypes<const Index, 1>::Tensor32Bit labels,
      const Index max_depth)
      : logits_(logits),
        sum_exp_logits_(sum_exp_logits),
        labels_(labels),
        max_depth_(max_depth) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<int, 2>& coords) const {
    const int batch = coords[0];
    const int depth = coords[1];
    const Index label = tensorflow::internal::SubtleMustCopy(labels_(batch));
    if (!FastBoundsCheck(label, max_depth_)) {
      return Eigen::NumTraits<T>::quiet_NaN();
    }
    return TF_PREDICT_FALSE(label == depth)
               ? (Eigen::numext::log(sum_exp_logits_(batch)) - logits_(coords))
               : T(0.0);
  };

 private:
  typename TTypes<const T, 2>::Tensor32Bit logits_;
  typename TTypes<const T, 1>::Tensor32Bit sum_exp_logits_;
  typename TTypes<const Index, 1>::Tensor32Bit labels_;
  const Index max_depth_;
};

// Generator for calculation of the sparse Xent gradient.
// This generator takes the exponentiated logits, their sums, and the label
// indices. For each minibatch entry, ignoring the batch index b, it calculates:
//
//   exp_logits[j] / sum_exp_logits - 1{ j == label }
//
// for j = 0 .. num_classes.
template <typename T, typename Index>
class SparseXentGradGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE SparseXentGradGenerator(
      typename TTypes<const T, 2>::Tensor32Bit exp_logits,
      typename TTypes<const T, 1>::Tensor32Bit sum_exp_logits,
      typename TTypes<const Index, 1>::Tensor32Bit labels,
      const Index max_depth)
      : exp_logits_(exp_logits),
        sum_exp_logits_(sum_exp_logits),
        labels_(labels),
        max_depth_(max_depth) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<int, 2>& coords) const {
    const int batch = coords[0];
    const int depth = coords[1];
    const Index label = tensorflow::internal::SubtleMustCopy(labels_(batch));
    if (!FastBoundsCheck(label, max_depth_)) {
      return Eigen::NumTraits<T>::quiet_NaN();
    }
    T subtract = TF_PREDICT_FALSE(depth == label) ? T(1.0) : T(0.0);
    return exp_logits_(coords) / sum_exp_logits_(batch) - subtract;
  };

 private:
  typename TTypes<const T, 2>::Tensor32Bit exp_logits_;
  typename TTypes<const T, 1>::Tensor32Bit sum_exp_logits_;
  typename TTypes<const Index, 1>::Tensor32Bit labels_;
  const Index max_depth_;
};

}  // namespace generator

namespace functor {
// TensorXXX functions are copied from lstm_ops modules
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

// Functor used by SparseXentOp to do the computations.
template <typename Device, typename T, typename Index>
struct SparseXentFunctor {
  // Computes Cross Entropy loss and backprop.
  //
  // logits: batch_size, num_classes.
  // labels: num_classes.
  // scratch: temporary tensor, dims: batch_size, 1
  // loss: output tensor for the loss, dims: batch_size.
  // backprop: output tensor for the backprop, dims: batch_size, num_classes.
  void operator()(OpKernelContext* ctx, const Device& d, typename TTypes<T>::ConstMatrix h, typename TTypes<Index>::ConstVec labels,
                  typename TTypes<T>::ConstMatrix w_y, typename TTypes<T>::ConstVec b_y, 
                  typename TTypes<T>::Matrix logits, typename TTypes<T>::Vec scratch, typename TTypes<T>::Matrix backprop,
                  typename TTypes<T>::Vec loss, typename TTypes<T>::Matrix h_grad,
                  typename TTypes<T>::Matrix dw_y, typename TTypes<T>::Vec db_y);
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RNN_SOFTMAXLOSS_HGRAD_OP_H_
