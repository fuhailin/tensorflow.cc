/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
// rnn ops, Vanalli RNN for now
// Author: Rock Zhuang
// Date  : Dec 20, 2018
// 

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/rnn_ops.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/eigen_activations.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define VERBOSE 1
#define TESTING 1

namespace tensorflow {
namespace functor {

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

// functor cont.
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

namespace {
// RowMajor, M(row, col) = *(M.elements + row * M.width + col)
template <typename T>
struct ConstMatrix
{
    int width;
    int height;
    const T *elements;
};

template <typename T>
struct Matrix
{
    int width;
    int height;
    T *elements;
};

//
template <typename T> 
__device__ T getElement(ConstMatrix<T> *A, int row, int col)
{
    return A->elements[row * A->width + col];
}

// 
template <typename T>
__device__ void setElement(Matrix<T> *A, int row, int col, T value)
{
    A->elements[row * A->width + col] = value;
}

// kernel of computing h, 1-D, one output scalar once
template <typename T>
__global__ void computeH_1D(const T *d1, int w1, int h1,
                           const T *d2, int w2, int h2,
                           const T *d3, int w3, int h3,
                           const T *d4, int w4, int h4,
                           const T *d5, int w5, int h5,
                           T *d6, int w6, int h6)
{
    ConstMatrix<T> w_xh_mat;
    w_xh_mat.height = h1;
    w_xh_mat.width = w1;
    w_xh_mat.elements = d1;

    ConstMatrix<T> w_hh_mat;
    w_hh_mat.height = h2;
    w_hh_mat.width = w2;
    w_hh_mat.elements = d2;

    ConstMatrix<T> x_mat;
    x_mat.height = h3;
    x_mat.width = w3;
    x_mat.elements = d3;

    ConstMatrix<T> h_prev_mat;
    h_prev_mat.height = h4;
    h_prev_mat.width = w4;
    h_prev_mat.elements = d4;

    ConstMatrix<T> b_h_mat;
    b_h_mat.height = h5;
    b_h_mat.width = w5;
    b_h_mat.elements = d5;

    Matrix<T> h_mat;
    h_mat.height = h6;
    h_mat.width = w6;
    h_mat.elements = d6;

    Eigen::internal::scalar_tanh_op<T> tanh_op;

    T w_xh__x = 0.0;
    T w_hh__h_prev = 0.0;
    T h_value = 0.0;

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = 0;

    // not in range
    if(row >= b_h_mat.height)
      return;

    // printf("--------------computeH: row:%d, col:%d\n", row, col);

    // w_xh[row] .* x[col]
    for (int i = 0; i < w_xh_mat.width; ++i)
    {
        w_xh__x += getElement(&w_xh_mat, row, i) * getElement(&x_mat, i, col);
    }

    // printf("--------------computeH: w_xh__x:%f\n", w_xh__x);

    // w_hh[row] * h_prev[col]
    for (int i = 0; i < w_hh_mat.width; ++i)
    {
        w_hh__h_prev += getElement(&w_hh_mat, row, i) * getElement(&h_prev_mat, i, col);
    }

    // printf("--------------computeH: w_hh__h_prev:%f\n", w_hh__h_prev);

    h_value = tanh_op(w_xh__x + w_hh__h_prev + getElement(&b_h_mat, row, col));

    setElement(&h_mat, row, col, h_value);
}

// kernel of computing h, 1-D, one output scalar once
template <typename T>
__global__ void computeYCExp_1D(const T *d1, int w1, int h1, 
                               const T *d2, int w2, int h2,
                               const T *d3, int w3, int h3,
                               T *d4, int w4, int h4)
{
    ConstMatrix<T> w_hy_mat;
    w_hy_mat.height = h1;
    w_hy_mat.width = w1;
    w_hy_mat.elements = d1;

    ConstMatrix<T> h_mat;
    h_mat.height = h2;
    h_mat.width = w2;
    h_mat.elements = d2;

    ConstMatrix<T> b_y_mat;
    b_y_mat.height = h3;
    b_y_mat.width = w3;
    b_y_mat.elements = d3;

    Matrix<T> y_c_exp_mat;
    y_c_exp_mat.height = h4;
    y_c_exp_mat.width = w4;
    y_c_exp_mat.elements = d4;

    Eigen::internal::scalar_exp_op<T> exp_op;

    T w_hy__h = 0.0;
    T y_c_exp_value = 0.0;

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = 0;

    // not in range
    if(row >= b_y_mat.height)
      return;

    // printf("--------------computeYCExp_1D: row:%d, col:%d\n", row, col);

    // w_hy[row] .* x[col]
    for (int i = 0; i < w_hy_mat.width; ++i)
    {
        w_hy__h += getElement(&w_hy_mat, row, i) * getElement(&h_mat, i, col);
    }

    // printf("--------------computeH: w_hh__h_prev:%f\n", w_hh__h_prev);

    y_c_exp_value = exp_op(w_hy__h + getElement(&b_y_mat, row, col));

    setElement(&y_c_exp_mat, row, col, y_c_exp_value);
}

// kernel of computing h, 1-D, one output scalar once
template <typename T>
__global__ void computeP_1D(const T *d1, int w1, int h1, 
                            T *sum_value,
                            T *d2, int w2, int h2)
{
    ConstMatrix<T> y_c_exp_mat;
    y_c_exp_mat.height = h1;
    y_c_exp_mat.width = w1;
    y_c_exp_mat.elements = d1;

    Matrix<T> p_mat;
    p_mat.height = h2;
    p_mat.width = w2;
    p_mat.elements = d2;

    T div_value = 0.0;

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = 0;

    // not in range
    if(row >= y_c_exp_mat.height)
      return;

    // printf("--------------computeP_1D: row:%d, col:%d\n", row, col);

    div_value = getElement(&y_c_exp_mat, row, col) / sum_value[0];

    setElement(&p_mat, row, col, div_value);
}

template <typename T>
void VanillaRNNCellFpropWithCUDA(
    const VanillaRNNCell& cell, OpKernelContext* ctx, const GPUDevice& d, const int64 t,
    typename TTypes<T>::ConstMatrix x,
    typename TTypes<T>::ConstScalar y,
    typename TTypes<T>::ConstMatrix h_prev,
    typename TTypes<T>::ConstMatrix w_xh,
    typename TTypes<T>::ConstMatrix w_hh,
    typename TTypes<T>::ConstMatrix w_hy,
    typename TTypes<T>::ConstMatrix b_h,
    typename TTypes<T>::ConstMatrix b_y,
    typename TTypes<T>::Matrix p_out, typename TTypes<T>::Matrix h_out, typename TTypes<T>::Scalar loss_out) {
  const cudaStream_t& cu_stream = GetCudaStream(ctx);

  T y_index;
  d.memcpyDeviceToHost(&y_index, y.data(), sizeof(T) * 1);

  {
  // 1D
  dim3 blockSize(32);
  dim3 gridSize((b_h.dimension(0) + blockSize.x - 1) / blockSize.x);

  // Corresponding Python code:
  //  h[t] = np.tanh(np.dot(self.W_xh, xhat[t]) + np.dot(self.W_hh, h[t-1]) + self.b_h)#find new hidden state
  computeH_1D<T><<<gridSize, blockSize>>>(w_xh.data(), w_xh.dimension(1), w_xh.dimension(0),
                                       x.data(), x.dimension(1), x.dimension(0),
                                       w_hh.data(), w_hh.dimension(1), w_hh.dimension(0),
                                       h_prev.data(), h_prev.dimension(1), h_prev.dimension(0),
                                       b_h.data(), b_h.dimension(1), b_h.dimension(0),
                                       h_out.data(), h_out.dimension(1), h_out.dimension(0));
  }

  //  yhat[t] = np.dot(self.W_hy, h[t]) + self.b_y#find unnormalized log probabilities for next chars
  Tensor y_c_exp_tensor;   // temporary variable y_c_exp
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({cell.input_size(), 1}), &y_c_exp_tensor));
  typename TTypes<T>::Matrix y_c_exp = y_c_exp_tensor.matrix<T>();

  {
  // 1D
  dim3 blockSize(32);
  dim3 gridSize((b_y.dimension(0) + blockSize.x - 1) / blockSize.x);
  computeYCExp_1D<T><<<gridSize, blockSize>>>(w_hy.data(), w_hy.dimension(1), w_hy.dimension(0),
                                       h_out.data(), h_out.dimension(1), h_out.dimension(0),
                                       b_y.data(), b_y.dimension(1), b_y.dimension(0),
                                       y_c_exp.data(), y_c_exp.dimension(1), y_c_exp.dimension(0));
  }

  //  p[t] = np.exp(yhat[t]) / np.sum(np.exp(yhat[t]))#find probabilities for next chars
  Eigen::array<Eigen::DenseIndex, 2> b_shape({1, 1});
  Eigen::array<Eigen::DenseIndex, 2> bcast({cell.input_size(), 1});
  p_out.device(d) = y_c_exp / y_c_exp.sum().reshape(b_shape).broadcast(bcast);

  // // Not used for now
  // Eigen::Tensor<T, 0, Eigen::RowMajor> sum_t;
  // sum_t.device(d) = y_c_exp.sum();
  // computeP_1D<T><<<gridSize, blockSize>>>(y_c_exp.data(), y_c_exp.dimension(1), y_c_exp.dimension(0),
  //                                      sum_t.data(),
  //                                      p_out.data(), p_out.dimension(1), p_out.dimension(0));
  
  // Corresponding Python code:
  //  loss += -np.log(p[t][y[t],0])#softmax (cross-entropy loss)
  //  
  loss_out.device(d) += -p_out.chip(static_cast<int>(y_index), 0).log();
}


// kernel of computing h, 1-D, one output scalar once
template <typename T>
__global__ void computeDY_1D(const T *d1, int w1, int h1, 
                            const int y_index,
                            T *d2, int w2, int h2)
{
    ConstMatrix<T> p_mat;
    p_mat.height = h1;
    p_mat.width = w1;
    p_mat.elements = d1;

    Matrix<T> dy_mat;
    dy_mat.height = h2;
    dy_mat.width = w2;
    dy_mat.elements = d2;

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = 0;

    // not in range
    if(row >= p_mat.height)
      return;

    // printf("--------------computeDY_1D: row:%d, col:%d\n", row, col);

    T out_value = getElement(&p_mat, row, col);

    if(row == y_index)
      out_value -= static_cast<T>(1.0f);

    setElement(&dy_mat, row, col, out_value);
}


// kernel of computing h, 1-D, one output scalar once
template <typename T>
__global__ void computeDhRaw_1D(const T *d1, int w1, int h1, 
                            const T *d2, int w2, int h2, 
                            T *d3, int w3, int h3)
{
    ConstMatrix<T> h_mat;
    h_mat.height = h1;
    h_mat.width = w1;
    h_mat.elements = d1;

    ConstMatrix<T> dh_mat;
    dh_mat.height = h2;
    dh_mat.width = w2;
    dh_mat.elements = d2;

    Matrix<T> dh_raw_mat;
    dh_raw_mat.height = h3;
    dh_raw_mat.width = w3;
    dh_raw_mat.elements = d3;

    T out_value = 0.0;

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = 0;

    // not in range
    if(row >= h_mat.height)
      return;

    // printf("--------------computeDhRaw_1D: row:%d, col:%d\n", row, col);

    T h_value = getElement(&h_mat, row, col);

    out_value = (static_cast<T>(1.0f) - h_value * h_value) * getElement(&dh_mat, row, col);

    setElement(&dh_raw_mat, row, col, out_value);
}

template <typename T>
void VanillaRNNCellBpropWithCUDA(
    const VanillaRNNCell& cell, OpKernelContext* ctx, const GPUDevice& d, const int64 t,
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
      typename TTypes<T>::Matrix d_b_y_out) {
  const cudaStream_t& cu_stream = GetCudaStream(ctx);

  T y_index;
  d.memcpyDeviceToHost(&y_index, y.data(), sizeof(T) * 1);

  // Python code:
  //   dy = np.copy(p[t])
  //   dy[y[t]] -= 1
  Tensor dy_tensor;   // temporary variable dy_tensor
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({p.dimension(0), p.dimension(1)}), &dy_tensor));
  typename TTypes<T>::Matrix dy = dy_tensor.matrix<T>();

  {
  // 1D
  dim3 blockSize(32);
  dim3 gridSize((p.dimension(0) + blockSize.x - 1) / blockSize.x);  

  computeDY_1D<T><<<gridSize, blockSize>>>(p.data(), p.dimension(1), p.dimension(0),
                                       static_cast<int>(y_index),
                                       dy.data(), dy.dimension(1), dy.dimension(0));

  }

  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) }; // for dot product
  const Eigen::array<Eigen::DenseIndex, 2> matrix_transpose({1, 0}); // For matrix transpose

  //   dW_hy += np.dot(dy, h[t].T)
  d_w_hy_out.device(d) += dy.contract(h.shuffle(matrix_transpose), product_dims);

  //   db_y += dy
  d_b_y_out.device(d) += dy;

  //   dh = np.dot(self.W_hy.T, dy) + dh_next
  Tensor dh_tensor;   // temporary variable dh_tensor
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({h.dimension(0), h.dimension(1)}), &dh_tensor));
  typename TTypes<T>::Matrix dh = dh_tensor.matrix<T>();
  dh.device(d) = w_hy.shuffle(matrix_transpose).contract(dy, product_dims) + dh_next;

  //   dh_raw = (1 - h[t]**2) * dh
  Tensor dh_raw_tensor;   // temporary variable dh_raw_tensor
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({h.dimension(0), h.dimension(1)}), &dh_raw_tensor));
  typename TTypes<T>::Matrix dh_raw = dh_raw_tensor.matrix<T>();

  {
  // 1D
  dim3 blockSize(32);
  dim3 gridSize((h.dimension(0) + blockSize.x - 1) / blockSize.x);  

  computeDhRaw_1D<T><<<gridSize, blockSize>>>(h.data(), h.dimension(1), h.dimension(0),
                                       dh.data(), dh.dimension(1), dh.dimension(0),
                                       dh_raw.data(), dh_raw.dimension(1), dh_raw.dimension(0));

  }

  //   dW_xh += np.dot(dh_raw, xhat[t].T)
  //   dW_hh += np.dot(dh_raw, h[t-1].T)
  //   db_h += dh_raw
  d_w_xh_out.device(d) += dh_raw.contract(x.shuffle(matrix_transpose), product_dims);
  d_w_hh_out.device(d) += dh_raw.contract(h_prev.shuffle(matrix_transpose), product_dims);
  d_b_h_out.device(d) += dh_raw;

  //   dh_next = np.dot(self.W_hh.T, dh_raw)
  dh_next.device(d) = w_hh.shuffle(matrix_transpose).contract(dh_raw, product_dims);  
}

}  // namespace

#define DEFINE_GPU_SPECS(T)                                                    \
  template struct TensorZero<GPUDevice, T>;                                    \
  template struct TensorUnalignedZero<GPUDevice, T>;                           \
  template struct TensorCopy<GPUDevice, T>;                                    \
  template struct TensorCopyUnaligned<GPUDevice, T>;                           \
  template struct TensorCopyToUnaligned<GPUDevice, T>;                         \
  template struct TensorAdd<GPUDevice, T>;                                     \
  template <>                                                                  \
  void VanillaRNNCellFprop<GPUDevice, T, true /* USE_CUBLAS */>::operator()(    \
      OpKernelContext* ctx, const GPUDevice& d, const int64 t,                      \
      typename TTypes<T>::ConstMatrix x,                                      \
      typename TTypes<T>::ConstScalar y,                                      \
      typename TTypes<T>::ConstMatrix h_prev,                                 \
      typename TTypes<T>::ConstMatrix w_xh,                                      \
      typename TTypes<T>::ConstMatrix w_hh,                                      \
      typename TTypes<T>::ConstMatrix w_hy,                                      \
      typename TTypes<T>::ConstMatrix b_h,                                       \
      typename TTypes<T>::ConstMatrix b_y,                                       \
      typename TTypes<T>::Matrix p_out, typename TTypes<T>::Matrix h_out,        \
      typename TTypes<T>::Scalar loss_out) {                                      \
    VanillaRNNCellFpropWithCUDA<T>(                                           \
        *this, ctx, d, t, x, y, h_prev, w_xh, w_hh, w_hy, b_h, b_y, p_out, h_out, loss_out);       \
  }                                                                            \
  template <>                                                                  \
  void VanillaRNNCellBprop<GPUDevice, T, true /* USE_CUBLAS */>::operator()(    \
      OpKernelContext* ctx, const GPUDevice& d, const int64 t,                      \
      typename TTypes<T>::ConstMatrix x,                                      \
      typename TTypes<T>::ConstScalar y,                                      \
      typename TTypes<T>::ConstMatrix p,                                 \
      typename TTypes<T>::ConstMatrix h,                                 \
      typename TTypes<T>::ConstMatrix w_hh,                                 \
      typename TTypes<T>::ConstMatrix w_hy,                              \
      typename TTypes<T>::ConstMatrix h_prev,                             \
      typename TTypes<T>::Matrix dh_next,                                \
      typename TTypes<T>::Matrix d_w_xh_out,                                      \
      typename TTypes<T>::Matrix d_w_hh_out,                                      \
      typename TTypes<T>::Matrix d_w_hy_out,                                      \
      typename TTypes<T>::Matrix d_b_h_out,                                       \
      typename TTypes<T>::Matrix d_b_y_out) {                                      \
    VanillaRNNCellBpropWithCUDA<T>(                                           \
        *this, ctx, d, t, x, y, p, h, w_hh, w_hy, h_prev, dh_next, d_w_xh_out, d_w_hh_out, d_w_hy_out, d_b_h_out, d_b_y_out); \
  }                                                                            \
  template struct VanillaRNNCellFprop<GPUDevice, T, true /* USE_CUBLAS */>;     \
  template struct VanillaRNNCellBprop<GPUDevice, T, true /* USE_CUBLAS */>;     

DEFINE_GPU_SPECS(float);
// DEFINE_GPU_SPECS(Eigen::half);
// DEFINE_GPU_SPECS(double);
#undef DEFINE_GPU_SPECS

}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
