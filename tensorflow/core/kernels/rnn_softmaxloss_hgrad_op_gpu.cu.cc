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
// to compute loss, h_grad, dw_y, db_y values.
// Ref: sparse_xent_op and lstm_ops modules
//
// Author: Rock Zhuang
// Date  : Jan 15, 2019
// 

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/rnn_softmaxloss_hgrad_op.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

struct FloatToHalf {
  __host__ __device__ EIGEN_STRONG_INLINE Eigen::half operator()(
      const float& x) const {
    return Eigen::half_impl::float_to_half_rtne(x);
  }
};

template <typename U, typename T>
__host__ __device__ EIGEN_STRONG_INLINE
    typename std::enable_if<!std::is_same<T, U>::value, U>::type
    strict_cast(T t);

template <typename U, typename T>
__host__ __device__ EIGEN_STRONG_INLINE
    typename std::enable_if<std::is_same<T, U>::value, U>::type
    strict_cast(T t) {
  return t;
}

template <>
__host__ __device__ EIGEN_STRONG_INLINE Eigen::half
strict_cast<Eigen::half, float>(float t) {
  return FloatToHalf()(t);
}

}  // namespace

// Partial specialization for a GPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
namespace functor {
template <typename T>
struct TensorZero<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat t) {
    t.device(d) = t.constant(strict_cast<T>(0.f));
  }
};

template <typename T>
struct TensorUnalignedZero<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::UnalignedFlat t) {
    t.device(d) = t.constant(strict_cast<T>(0.f));
  }
};

template <typename T, typename Index>
struct SparseXentFunctor<GPUDevice, T, Index> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d, typename TTypes<T>::ConstMatrix h, typename TTypes<Index>::ConstVec labels,
                  typename TTypes<T>::ConstMatrix w_y, typename TTypes<T>::ConstVec b_y, 
                  typename TTypes<T>::Matrix logits, typename TTypes<T>::Vec scratch, typename TTypes<T>::Matrix backprop,
                  typename TTypes<T>::Vec loss, typename TTypes<T>::Matrix h_grad,
                  typename TTypes<T>::Matrix dw_y, typename TTypes<T>::Vec db_y) {
    // const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = h.dimension(0);
    const int num_classes = w_y.dimension(0);

    // Get logits for a batch:
    //  y_bar = np.dot(W_y, h) + b_y

    // CPU Version
    // Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    // contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 1);
    // To32Bit(logits).device(d) = To32Bit(h).contract(w_y, contract_pairs) + To32Bit(b_y).reshape(b_y_shape).broadcast(broadcast_shape); // batch logits with shape (batch_size, input_size)

    Eigen::array<Eigen::DenseIndex, 2> b_y_shape({1, b_y.dimensions()[0]});
    Eigen::array<Eigen::DenseIndex, 2> broadcast_shape({batch_size, 1});
    logits.device(d) = b_y.reshape(b_y_shape).broadcast(broadcast_shape); // batch logits with shape (batch_size, input_size)

    TensorBlasGemm<GPUDevice, T, true>::compute(
        ctx, d, false, true, 1.f, h, w_y, 1.f, logits);
    
// These arrays are used to reduce along the class dimension, and broadcast
// the resulting value to all classes.
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 1> along_class;
    along_class[0] = kClassDim;
    Eigen::array<int, 1> batch_only;
    batch_only[0] = batch_size;
    Eigen::array<int, 2> batch_by_one;
    batch_by_one[0] = batch_size;
    batch_by_one[1] = 1;
    Eigen::array<int, 2> one_by_class;
    one_by_class[0] = 1;
    one_by_class[1] = num_classes;
#else
    Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
    Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
    batch_by_one.set(0, batch_size);
    Eigen::IndexList<int> batch_only;
    batch_only.set(0, batch_size);
    Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
    one_by_class.set(1, num_classes);
#endif

    // scratch = max_logits along classes.
    To32Bit(scratch).device(d) = To32Bit(logits).maximum(along_class);

    // backprop = logits - max_logits.
    To32Bit(backprop).device(d) =
        To32Bit(logits) -
        To32Bit(scratch).reshape(batch_by_one).broadcast(one_by_class);

    // scratch = sum(exp(logits - max_logits)) along classes.
    To32Bit(scratch).device(d) = To32Bit(backprop).exp().sum(along_class);

    //  sum(-labels *
    //     ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
    //  along classes
    generator::SparseXentLossGenerator<T, Index> sparse_xent_loss_gen(
        sparse_xent_helpers::To32BitConst<T>(backprop),
        sparse_xent_helpers::To32BitConst<T>(scratch), To32Bit(labels),
        backprop.dimension(1) /* max_depth */);
    To32Bit(loss).device(d) =
        To32Bit(backprop).generate(sparse_xent_loss_gen).sum(along_class);

    // backprop: prob - labels, where
    //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
    To32Bit(backprop).device(d) = To32Bit(backprop).exp();
    generator::SparseXentGradGenerator<T, Index> sparse_xent_grad_gen(
        sparse_xent_helpers::To32BitConst<T>(backprop),
        sparse_xent_helpers::To32BitConst<T>(scratch), To32Bit(labels),
        backprop.dimension(1) /* max_depth */);
    To32Bit(backprop).device(d) =
        To32Bit(backprop).generate(sparse_xent_grad_gen);

    // dW_y += np.dot(dy, h.T), for a batch
    
    // CPU Version
    // Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs2;
    // contract_pairs2[0] = Eigen::IndexPair<Eigen::DenseIndex>(0, 0);
    // To32Bit(dw_y).device(d) += To32Bit(backprop).contract(h, contract_pairs2);

    typename TTypes<T>::ConstMatrix const_backprop(backprop.data(), backprop.dimensions());
    TensorBlasGemm<GPUDevice, T, true>::compute(
        ctx, d, true, false, 1.f, const_backprop, h, 1.f, dw_y);

    // db_y += dy, for a batch
    db_y.device(d) += backprop.sum(Eigen::array<int, 1>({0}));

    // dh = np.dot(W_y.T, dy), for a batch

    // CPU Version
    // Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs3;
    // contract_pairs3[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
    // To32Bit(h_grad).device(d) = To32Bit(backprop).contract(To32Bit(w_y), contract_pairs3);
    TensorBlasGemm<GPUDevice, T, true>::compute(
        ctx, d, false, false, 1.f, const_backprop, w_y, 0.f, h_grad);

  }
};

}  // end namespace functor

// Instantiate the GPU implementation for float.
#define REGISTER(Index)                                                      \
  template struct functor::SparseXentFunctor<GPUDevice, float, Index>;       \
  template class generator::SparseXentGradGenerator<float, Index>;           \
  template struct functor::SparseXentFunctor<GPUDevice, Eigen::half, Index>; \
  template class generator::SparseXentGradGenerator<Eigen::half, Index>;

REGISTER(int32)
REGISTER(int64)
#undef REGISTER

#define REGISTER_TENSOR_FUNCTOR(T)                                                      \
  template struct functor::TensorZero<GPUDevice, T>;                                    \
  template struct functor::TensorUnalignedZero<GPUDevice, T>;                           \
  template struct functor::TensorCopy<GPUDevice, T>;                                    \
  template struct functor::TensorCopyUnaligned<GPUDevice, T>;                           \
  template struct functor::TensorCopyToUnaligned<GPUDevice, T>;                         \
  template struct functor::TensorAdd<GPUDevice, T>; 

REGISTER_TENSOR_FUNCTOR(float)
REGISTER_TENSOR_FUNCTOR(Eigen::half)
#undef REGISTER_TENSOR_FUNCTOR

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
