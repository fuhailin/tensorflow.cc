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
// TODO: to comply with @tf_export("RegisterGradient")
//
// Author: Rock Zhuang
// Date  : Dec 20, 2018
// 

//
// Ref: lstm_ops.cc
// Eigen Tensor Ref: https://github.com/eigenteam/eigen-git-mirror/tree/master/unsupported/Eigen/CXX11/src/Tensor
//   and https://eigen.tuxfamily.org/dox-devel/unsupported/eigen_tensors.html#title0
// 

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

// #define VERBOSE 1
// #define TESTING 1

#define EIGEN_USE_THREADS

#include <memory>
#include <vector>
#include <algorithm>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/kernels/rkz_rnn_ops.h"

namespace tensorflow {

// typedef Eigen::ThreadPoolDevice CPUDevice;
// typedef Eigen::GpuDevice GPUDevice;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

// Forward pass for one input character of the input sequence
template <typename T>
void VanillaRNNCellFpropWithEigen(
    const VanillaRNNCell& cell, OpKernelContext* ctx, const CPUDevice& d, const int64 t,
    typename TTypes<T>::ConstMatrix x,
    typename TTypes<T>::ConstScalar y,
    typename TTypes<T>::ConstMatrix h_prev,
    typename TTypes<T>::ConstMatrix w_xh,
    typename TTypes<T>::ConstMatrix w_hh,
    typename TTypes<T>::ConstMatrix w_hy,
    typename TTypes<T>::ConstMatrix b_h,
    typename TTypes<T>::ConstMatrix b_y,
    typename TTypes<T>::Matrix p_out, typename TTypes<T>::Matrix h_out, typename TTypes<T>::Scalar loss_out) {

  // Note that that Tensorflow uses Eigen::RowMajor, don't mix it with Eigen::ColMajor

  int y_index = static_cast<int>(y());

#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "---------------------------------------------------------sequence number:" << std::endl << t;
  LOG(INFO) << __FUNCTION__ << "----------------------------x:" << std::endl << x;
  LOG(INFO) << __FUNCTION__ << "----------------------------y:" << std::endl << y;
  LOG(INFO) << __FUNCTION__ << "----------------------------y_index:" << std::endl << y_index;
  LOG(INFO) << __FUNCTION__ << "----------------------------h_prev:" << std::endl << h_prev;
  LOG(INFO) << __FUNCTION__ << "----------------------------w_xh:" << std::endl << w_xh;
  LOG(INFO) << __FUNCTION__ << "----------------------------w_hh:" << std::endl << w_hh;
  LOG(INFO) << __FUNCTION__ << "----------------------------b_h:" << std::endl << b_h;
#endif

  // LOG(INFO) << __FUNCTION__ << "----------------------------y_index:" << std::endl << y_index;

  // Corresponding Python code:
  //  h[t] = np.tanh(np.dot(self.W_xh, xhat[t]) + np.dot(self.W_hh, h[t-1]) + self.b_h)#find new hidden state
  //  yhat[t] = np.dot(self.W_hy, h[t]) + self.b_y#find unnormalized log probabilities for next chars
  //  p[t] = np.exp(yhat[t]) / np.sum(np.exp(yhat[t]))#find probabilities for next chars
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) }; // for dot product
  auto h = (w_xh.contract(x, product_dims) + w_hh.contract(h_prev, product_dims) + b_h).tanh();
  auto y_c = w_hy.contract(h, product_dims) + b_y;
  auto y_c_exp = y_c.exp();

  Eigen::array<Eigen::DenseIndex, 2> b_shape({1, 1});
  Eigen::array<Eigen::DenseIndex, 2> bcast({cell.input_size(), 1});
  auto p = y_c_exp / y_c_exp.sum().reshape(b_shape).broadcast(bcast);

#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "----------------------------softmax h:" << std::endl << h;
  LOG(INFO) << __FUNCTION__ << "----------------------------softmax y_c:" << std::endl << y_c;
  LOG(INFO) << __FUNCTION__ << "----------------------------softmax y_c_exp:" << std::endl << y_c_exp;
  LOG(INFO) << __FUNCTION__ << "----------------------------softmax p:" << std::endl << p;
#endif

  // Corresponding Python code:
  //  loss += -np.log(p[t][y[t],0])#softmax (cross-entropy loss)
  // 
  auto py = p.chip(y_index, 0);
  loss_out.device(d) += -py.log();

  p_out.device(d) = p;
  h_out.device(d) = h;

#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "----------------------------py:" << std::endl << py;
  LOG(INFO) << __FUNCTION__ << "----------------------------loss: " << std::endl << -py.log() << ", cumulated: " << loss_out;
#endif


#ifdef TESTING

  {
    // Test
    Eigen::Tensor<int, 2> a(2, 3);
    a.setValues({{1, 2, 3}, {6, 5, 4}});

    Eigen::Tensor<int, 2> b(3, 2);
    b.setValues({{1, 2}, {4, 5}, {5, 6}});

    const Eigen::array<Eigen::DenseIndex, 2> matrix_transpose({1, 0});
    Eigen::Tensor<int, 2> output = a.shuffle(matrix_transpose); // transpose

    // LOG(INFO) << __FUNCTION__ << "----------------------------output:" << std::endl << output;

    // auto c = a * b;
    // LOG(INFO) << __FUNCTION__ << "----------------------------output:" << std::endl << c;
  }

  {
    // Create 2 matrices using tensors of rank 2
    Eigen::Tensor<int, 2> a(2, 3);
    a.setValues({{1, 2, 3}, {6, 5, 4}});
    Eigen::Tensor<int, 2> b(3, 2);
    b.setValues({{1, 2}, {4, 5}, {5, 6}});

    // Compute the traditional matrix product
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    Eigen::Tensor<int, 2> AB = a.contract(b, product_dims);
    // LOG(INFO) << __FUNCTION__ << "----------------------------contract:" << std::endl << AB;

  }

  {
    Eigen::Tensor<int, 2> a(2, 3);
    a.setValues({{1, 2, 3}, {6, 5, 4}});
    auto b = a.sum(Eigen::array<int, 2>({2, 3}));
    // LOG(INFO) << __FUNCTION__ << "----------------------------sum:" << std::endl << b;
    
    Eigen::array<int, 2> bcast({3, 2});
    // Eigen::Tensor<int, 2> b = a.broadcast(bcast);
  }

#endif
}

// Backward pass for one input character of the input sequence
template <typename T>
void VanillaRNNCellBpropWithEigen(
    const VanillaRNNCell& cell, OpKernelContext* ctx, const CPUDevice& d, const int64 t,
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

  // Note that that Tensorflow uses Eigen::RowMajor, don't mix it with Eigen::ColMajor

  // Get y index
  int y_index = static_cast<int>(y());

  // Python code:
  //   dy = np.copy(p[t])
  //   dy[y[t]] -= 1
  int p_data_len = p.dimension(0) * p.dimension(1);
  float dy_data[p_data_len];
  std::copy_n(p.data(), p_data_len, dy_data);
  dy_data[y_index] -= 1.0f;
  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> dy(dy_data, p.dimension(0), p.dimension(1));

#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "---------------------------------------------------------sequence number:" << std::endl << t;
  LOG(INFO) << __FUNCTION__ << "----------------------------x:" << std::endl << x;
  LOG(INFO) << __FUNCTION__ << "----------------------------y:" << std::endl << y;
  LOG(INFO) << __FUNCTION__ << "----------------------------y_index:" << std::endl << y_index;
  LOG(INFO) << __FUNCTION__ << "----------------------------p:" << std::endl << p;
  LOG(INFO) << __FUNCTION__ << "----------------------------h:" << std::endl << h;
  LOG(INFO) << __FUNCTION__ << "----------------------------h_prev:" << std::endl << h_prev;
  LOG(INFO) << __FUNCTION__ << "----------------------------dy:" << std::endl << dy;
  LOG(INFO) << __FUNCTION__ << "----------------------------input dh_next:" << std::endl << dh_next;
#endif

  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) }; // for dot product
  const Eigen::array<Eigen::DenseIndex, 2> matrix_transpose({1, 0}); // For matrix transpose

  //   dW_hy += np.dot(dy, h[t].T)
  auto h_t = h.shuffle(matrix_transpose);
  d_w_hy_out.device(d) += dy.contract(h_t, product_dims);
#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "----------------------------h_transpose:" << std::endl << h_t;
  LOG(INFO) << __FUNCTION__ << "----------------------------dw_hy:" << std::endl << d_w_hy_out;
#endif

  //   db_y += dy
  d_b_y_out.device(d) += dy;

  //   dh = np.dot(self.W_hy.T, dy) + dh_next
  auto w_hy_t = w_hy.shuffle(matrix_transpose);
  auto dh = w_hy_t.contract(dy, product_dims) + dh_next;

  //   dh_raw = (1 - h[t]**2) * dh
  Eigen::Tensor<float, 2, Eigen::RowMajor> one_h(h.dimension(0), h.dimension(1));
  one_h.setConstant(1.0f);
  Eigen::Tensor<float, 2, Eigen::RowMajor> dh_raw = (one_h - h.pow(2)) * dh;

#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "----------------------------dh_raw:" << std::endl << dh_raw;
#endif

  //   dW_xh += np.dot(dh_raw, xhat[t].T)
  //   dW_hh += np.dot(dh_raw, h[t-1].T)
  //   db_h += dh_raw
  d_w_xh_out.device(d) += dh_raw.contract(x.shuffle(matrix_transpose), product_dims);
  d_w_hh_out.device(d) += dh_raw.contract(h_prev.shuffle(matrix_transpose), product_dims);
  d_b_h_out.device(d) += dh_raw;

  //   dh_next = np.dot(self.W_hh.T, dh_raw)
  dh_next.device(d) = w_hh.shuffle(matrix_transpose).contract(dh_raw, product_dims);
#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "----------------------------d_w_xh_out:" << std::endl << d_w_xh_out;

  LOG(INFO) << __FUNCTION__ << "----------------------------updated dh_next:" << std::endl << dh_next;
#endif


#ifdef TESTING
  {// Matrix Transpose
    int storage[128];  // 2 x 4 x 2 x 8 = 128
    Eigen::TensorMap<Eigen::Tensor<int, 4>> t_4d(storage, 2, 4, 2, 8);

    float s2[12];  
    // Eigen::TensorMap<Eigen::Tensor<const float, 2>> h(s2, 3, 4);
    Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> h(s2, 3, 4);

    const Eigen::array<Eigen::DenseIndex, 2> matrix_transpose({1, 0});
    Eigen::Tensor<float, 2, Eigen::RowMajor> h_t = h.shuffle(matrix_transpose);
  #ifdef VERBOSE
    // LOG(INFO) << __FUNCTION__ << "----------------------------h_t:" << std::endl << h_t;
  #endif
  }

  { // ColMajor
    // Create 2 matrices using tensors of rank 2
    Eigen::Tensor<int, 2, Eigen::ColMajor> a(2, 1);
    a.setValues({{1}, {2}});
    Eigen::Tensor<int, 2, Eigen::ColMajor> b(1, 3);
    b.setValues({{1, 2, 3}});
    // LOG(INFO) << __FUNCTION__ << "----------------------------contract ColMajor a:" << std::endl << a;
    // LOG(INFO) << __FUNCTION__ << "----------------------------contract ColMajor b:" << std::endl << b;

    // Compute the traditional matrix product
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    Eigen::Tensor<int, 2, Eigen::ColMajor> AB = a.contract(b, product_dims);
    // LOG(INFO) << __FUNCTION__ << "----------------------------contract ColMajor:" << std::endl << AB;

    float storage[2];  
    storage[0] = 2,
    storage[1] = 3;
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> t_2d(storage, 2, 1);
    auto pow_result = t_2d.pow(2);
    LOG(INFO) << __FUNCTION__ << "----------------------------pow_result:" << std::endl << pow_result;

    float storage2[2];  
    storage2[0] = 3,
    storage2[1] = 4;
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> t_2d2(storage2, 2, 1);
    auto mul_result = t_2d * t_2d2;
    LOG(INFO) << __FUNCTION__ << "----------------------------mul_result:" << std::endl << mul_result;

  }

  { // RowMajor
    // Create 2 matrices using tensors of rank 2
    Eigen::Tensor<int, 2, Eigen::RowMajor> a(2, 1);
    a.setValues({{1}, {2}});
    Eigen::Tensor<int, 2, Eigen::RowMajor> b(1, 3);
    b.setValues({{1, 2, 3}});
    // LOG(INFO) << __FUNCTION__ << "----------------------------contract a:" << std::endl << a;
    // LOG(INFO) << __FUNCTION__ << "----------------------------contract b:" << std::endl << b;

    // Compute the traditional matrix product
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    Eigen::Tensor<int, 2, Eigen::RowMajor> AB = a.contract(b, product_dims);
    // LOG(INFO) << __FUNCTION__ << "----------------------------contract:" << std::endl << AB;

  }

  {
    // Eigen::Tensor<float, 1, Eigen::RowMajor> lr(1);
    // lr.setValues({0.01f});

    // Eigen::Tensor<float, 1, Eigen::RowMajor> grad(2);
    // grad.setValues({0.5f, 0.0f});

    // auto add_result = grad + 0.5f;
    // LOG(INFO) << __FUNCTION__ << "----------------------------add_result test:" << std::endl << add_result;

    // auto accum = grad.square();
    // LOG(INFO) << __FUNCTION__ << "----------------------------accum test:" << std::endl << accum;

    // LOG(INFO) << __FUNCTION__ << "----------------------------accum.rsqrt() test:" << std::endl << accum.rsqrt();

    // auto var = grad * lr * (accum + 1e-8f).rsqrt();
    // LOG(INFO) << __FUNCTION__ << "----------------------------var test:" << std::endl << var;
  }
#endif
}

// CPUDevice Instatiation of template class
#define DEFINE_CPU_SPECS(T)                                                   \
  template <>                                                                 \
  void VanillaRNNCellFprop<CPUDevice, T, false /* USE_CUBLAS */>::operator()(  \
      OpKernelContext* ctx, const CPUDevice& d, const int64 t,                      \
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
    VanillaRNNCellFpropWithEigen<T>(                                           \
        *this, ctx, d, t, x, y, h_prev, w_xh, w_hh, w_hy, b_h, b_y, p_out, h_out, loss_out);       \
  }                                                                           \
  template <>                                                                 \
  void VanillaRNNCellBprop<CPUDevice, T, false /* USE_CUBLAS */>::operator()(  \
      OpKernelContext* ctx, const CPUDevice& d, const int64 t,                      \
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
    VanillaRNNCellBpropWithEigen<T>(                                           \
        *this, ctx, d, t, x, y, p, h, w_hh, w_hy, h_prev, dh_next, d_w_xh_out, d_w_hh_out, d_w_hy_out, d_b_h_out, d_b_y_out); \
  }                                                                           \
  template struct VanillaRNNCellFprop<CPUDevice, T, false /* USE_CUBLAS */>;  \
  template struct VanillaRNNCellBprop<CPUDevice, T, false /* USE_CUBLAS */>; 

DEFINE_CPU_SPECS(float);
#undef DEFINE_CPU_SPECS

}  // namespace functor

namespace {

// This helper class can be used to access timeslices of a 3D tensor. If a slice
// happens to be unaligned (usually because both batch size and number of cells
// are odd - this isn't common) this involves overhead, since data needs to be
// copied. However, if all slices are aligned, the bits aren't copied. In the
// cases where copying is needed, the outputs have to be recopied back.
// At the end of each time step you should call FinishTimeStep which does this,
// and also allows for reuse of temporary tensors.
template <typename Device, typename T>
class SliceHelper {
 public:
  explicit SliceHelper(OpKernelContext* ctx)
      : ctx_(ctx), device_(ctx_->eigen_device<Device>()) {}

  ~SliceHelper() {
    CHECK(copy_out_.empty());
    for (const auto& entry : pool_) {
      CHECK(!entry.second.second);  // nothing is in use
    }
  }

  // Slice through an input tensor. This may copy unaligned slices, but no
  // copying back will be done at the end.
  const Tensor InputSlice(const Tensor& t, int pos, const string& name) {
    Tensor res = UnalignedSlice(t, pos);
    if (res.IsAligned()) {
      return res;
    } else {
      return AlignTensor(res, name);
    }
  }

  // Slice through an output tensor. This may copy unaligned slices, and
  // schedule copying back on destruction.
  Tensor OutputSlice(Tensor* t, int pos, const string& name) {
    Tensor res = UnalignedSlice(*t, pos);
    if (res.IsAligned()) {
      return res;
    } else {
      Tensor aligned = AlignTensor(res, name);
      copy_out_.emplace_back(res, aligned);
      return aligned;
    }
  }

  void FinishTimeStep() {
    for (const auto& p : copy_out_) {
      const Tensor& aligned = p.second;
      Tensor original = p.first;
      // Copy from aligned back to original.
      functor::TensorCopyToUnaligned<Device, T>()(device_, aligned.flat<T>(),
                                                  original.unaligned_flat<T>());
    }
    copy_out_.clear();
    // Mark all entries as not in use.
    for (auto& entry : pool_) {
      entry.second.second = false;
    }
  }

 private:
  // Return a slice at position 'pos'. Result may be unaligned. The resulting
  // tensor always shares data with the source tensor.
  Tensor UnalignedSlice(const Tensor& t, int pos) const {
    Tensor res;
    // CHECK should never fail here, since the number of elements must match
    CHECK(res.CopyFrom(t.Slice(pos, pos + 1), {t.dim_size(1), t.dim_size(2)}));
    return res;
  }

  // Assumes input is not aligned, creates a temporary aligned tensor of the
  // same shape and copies the original tensor's content into it.
  Tensor AlignTensor(const Tensor& t, const string& name) {
    VLOG(1) << "AlignTensor called for " << name << ", shape "
            << t.shape().DebugString()
            << ". This is unnecessary copying. Consider using shapes with even "
            << "sizes";
    Tensor aligned;
    auto found = pool_.find(name);
    if (found != pool_.end()) {  // found in pool
      CHECK(!found->second.second) << "Tensor " << name << " is in use";
      found->second.second = true;  // mark in use
      aligned = found->second.first;
      CHECK(aligned.shape().IsSameSize(t.shape()));
      CHECK_EQ(aligned.dtype(), t.dtype());
    } else {  // allocate a new temporary tensor
      TF_CHECK_OK(ctx_->allocate_temp(t.dtype(), t.shape(), &aligned));
      pool_.emplace(name, std::make_pair(aligned, true));
    }
    functor::TensorCopyUnaligned<Device, T>()(device_, t.unaligned_flat<T>(),
                                              aligned.flat<T>());
    return aligned;
  }

  // Tensors to be copied.
  std::vector<std::pair<Tensor, const Tensor>> copy_out_;
  // A pool of pre-allocated temporary tensors, with an indicator for whether
  // it's in use.
  std::map<string, std::pair<Tensor, bool>> pool_;
  // Op context
  OpKernelContext* ctx_ = nullptr;
  // Device
  const Device& device_;
};

}  // namespace

class TensorTestHelper {
 public:
  // This is an operation that can be done by VariableOp.
  static void set_shape(Tensor* t, const TensorShape& s) { t->set_shape(s); }
};

template <typename Device, typename T, bool USE_CUBLAS>
class VanillaRNNOp : public OpKernel {
 public:
  explicit VanillaRNNOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hidsize", &hidsize));
  }

  void Compute(OpKernelContext* ctx) override {
    // check the shape of x
    const Tensor* x_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));
    OP_REQUIRES(ctx, x_tensor->dims() == 3, errors::InvalidArgument("x must be 3D"));
    const int64 seq_length = x_tensor->dim_size(0);
    const int64 insize = x_tensor->dim_size(1);

    // check the shape of y
    const Tensor* y_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("y", &y_tensor));
    OP_REQUIRES(ctx, y_tensor->dims() == 1, errors::InvalidArgument("y must be 1D"));

    // check the shape of h_prev
    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));
    OP_REQUIRES(ctx, h_prev_tensor->dims() == 2,
                errors::InvalidArgument("h_prev must be 2D"));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == hidsize,
                errors::InvalidArgument("h_prev.dims(0) != hidsize: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        hidsize));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == 1,
                errors::InvalidArgument(
                    "h_prev.dims(1) != 1: ", h_prev_tensor->dim_size(1),
                    " vs. ", 1));

    // w_xh_tensor
    const Tensor* w_xh_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_xh", &w_xh_tensor));
    OP_REQUIRES(ctx, w_xh_tensor->dims() == 2,
                errors::InvalidArgument("w_xh must be 2D"));
    OP_REQUIRES(ctx, w_xh_tensor->dim_size(0) == hidsize,
                errors::InvalidArgument("w_xh.dims(0) != hidsize: ",
                                        w_xh_tensor->dim_size(0), " vs. ",
                                        hidsize));
    OP_REQUIRES(ctx, w_xh_tensor->dim_size(1) == insize,
                errors::InvalidArgument(
                    "w_xh.dims(1) != insize: ", w_xh_tensor->dim_size(1),
                    " vs. ", insize));

    // w_hh_tensor
    const Tensor* w_hh_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_hh", &w_hh_tensor));
    OP_REQUIRES(ctx, w_hh_tensor->dims() == 2,
                errors::InvalidArgument("w_hh must be 2D"));
    OP_REQUIRES(ctx, w_hh_tensor->dim_size(0) == hidsize,
                errors::InvalidArgument("w_hh.dims(0) != hidsize: ",
                                        w_hh_tensor->dim_size(0), " vs. ",
                                        hidsize));
    OP_REQUIRES(ctx, w_hh_tensor->dim_size(1) == hidsize,
                errors::InvalidArgument(
                    "w_hh.dims(1) != hidsize: ", w_hh_tensor->dim_size(1),
                    " vs. ", hidsize));

    // w_hy_tensor
    const Tensor* w_hy_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_hy", &w_hy_tensor));
    OP_REQUIRES(ctx, w_hy_tensor->dims() == 2,
                errors::InvalidArgument("w_hy must be 2D"));
    OP_REQUIRES(ctx, w_hy_tensor->dim_size(0) == insize,
                errors::InvalidArgument("w_hy.dims(0) != insize: ",
                                        w_hy_tensor->dim_size(0), " vs. ",
                                        insize));
    OP_REQUIRES(ctx, w_hy_tensor->dim_size(1) == hidsize,
                errors::InvalidArgument(
                    "w_hy.dims(1) != hidsize: ", w_hy_tensor->dim_size(1),
                    " vs. ", hidsize));

    // b_h_tensor
    const Tensor* b_h_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_h", &b_h_tensor));
    OP_REQUIRES(ctx, b_h_tensor->dims() == 2,
                errors::InvalidArgument("b_h must be 2D"));
    OP_REQUIRES(ctx, b_h_tensor->dim_size(0) == hidsize,
                errors::InvalidArgument("b_h.dims(0) != hidsize: ",
                                        b_h_tensor->dim_size(0), " vs. ",
                                        hidsize));
    OP_REQUIRES(ctx, b_h_tensor->dim_size(1) == 1,
                errors::InvalidArgument(
                    "b_h.dims(1) != 1: ", b_h_tensor->dim_size(1),
                    " vs. ", 1));

    // b_y_tensor
    const Tensor* b_y_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_y", &b_y_tensor));
    OP_REQUIRES(ctx, b_y_tensor->dims() == 2,
                errors::InvalidArgument("b_y must be 2D"));
    OP_REQUIRES(ctx, b_y_tensor->dim_size(0) == insize,
                errors::InvalidArgument("b_y.dims(0) != insize: ",
                                        b_y_tensor->dim_size(0), " vs. ",
                                        insize));
    OP_REQUIRES(ctx, b_y_tensor->dim_size(1) == 1,
                errors::InvalidArgument(
                    "b_y.dims(1) != 1: ", b_y_tensor->dim_size(1),
                    " vs. ", 1));

    // set shape of outputs
    TensorShape p_shape({seq_length, insize, 1});
    Tensor* p_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("p", p_shape, &p_out));

    TensorShape hid_shape({seq_length, hidsize, 1});
    Tensor* h_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("h", hid_shape, &h_out));

    TensorShape loss_shape({});
    Tensor* loss_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", loss_shape, &loss_tensor));

    // 
    const Device& device = ctx->eigen_device<Device>();

    functor::TensorZero<Device, T>()(device, p_out->flat<T>());
    functor::TensorZero<Device, T>()(device, h_out->flat<T>());
    functor::TensorZero<Device, T>()(device, loss_tensor->flat<T>());

    SliceHelper<Device, T> slicer(ctx);

    const Tensor *tmp_h_prev = h_prev_tensor;
    
    for (int64 t = 0; t < seq_length; ++t) {
      const Tensor x_t_tensor = slicer.InputSlice(*x_tensor, t, "x");

      Tensor p_tensor = slicer.OutputSlice(p_out, t, "p_out");
      Tensor h_tensor = slicer.OutputSlice(h_out, t, "h_out");

      // 
      if(t > 0) {
        Tensor h_t_1_tensor = slicer.OutputSlice(h_out, t - 1, "h_out");

        tmp_h_prev = &h_t_1_tensor;
      }

      // y
      typename TTypes<T>::ConstScalar y_t(&(y_tensor->vec<T>()(t)), 0);

      functor::VanillaRNNCellFprop<Device, T, USE_CUBLAS>(seq_length, insize, hidsize)(
          ctx, device, t,
          x_t_tensor.matrix<T>(), y_t, tmp_h_prev->matrix<T>(), 
          w_xh_tensor->matrix<T>(), w_hh_tensor->matrix<T>(), w_hy_tensor->matrix<T>(),
          b_h_tensor->matrix<T>(), b_y_tensor->matrix<T>(), 
          p_tensor.matrix<T>(), h_tensor.matrix<T>(), loss_tensor->scalar<T>());

// #ifdef VERBOSE
//   LOG(INFO) << __FUNCTION__ << "----------------------------loss cumulated:" << std::endl << loss_tensor->scalar<T>()();
// #endif
      slicer.FinishTimeStep();
    }


// #ifdef VERBOSE
//   Tensor h_tensorxxx = slicer.OutputSlice(h_out, seq_length - 1, "h_out");

//   LOG(INFO) << __FUNCTION__ << "----------------------------h_tensor xxx 111:" << std::endl << h_tensorxxx.matrix<T>();
// #endif
  }

 private:
  int32 hidsize;
};

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("VanillaRNN").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      VanillaRNNOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                \
  template <>                                                              \
  void VanillaRNNCellFprop<GPUDevice, T, true>::operator()(                 \
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
      typename TTypes<T>::Scalar loss_out);                                      \
                                                                           \
  extern template struct VanillaRNNCellFprop<GPUDevice, T, true>;

DECLARE_GPU_SPEC(float);
// DECLARE_GPU_SPEC(Eigen::half);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

#define REGISTER_GPU_KERNEL(T)                           \
  REGISTER_KERNEL_BUILDER(Name("VanillaRNN")             \
                              .Device(DEVICE_GPU)        \
                              .HostMemory("y")           \
                              .TypeConstraint<T>("T"),   \
                          VanillaRNNOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
// REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

template <typename Device, typename T, bool USE_CUBLAS>
class VanillaRNNGradOp : public OpKernel {
 public:
  explicit VanillaRNNGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hidsize", &hidsize));
  }

  void Compute(OpKernelContext* ctx) override {
    // check the shape of x
    const Tensor* x_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));
    OP_REQUIRES(ctx, x_tensor->dims() == 3, errors::InvalidArgument("x must be 3D"));
    const int64 seq_length = x_tensor->dim_size(0);
    const int64 insize = x_tensor->dim_size(1);

    // check the shape of y
    const Tensor* y_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("y", &y_tensor));
    OP_REQUIRES(ctx, y_tensor->dims() == 1, errors::InvalidArgument("y must be 1D"));

    // check the shape of p
    const Tensor* p_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("p", &p_tensor));
    OP_REQUIRES(ctx, p_tensor->dims() == 3,
                errors::InvalidArgument("p must be 3D"));
    OP_REQUIRES(ctx, p_tensor->dim_size(0) == seq_length,
                errors::InvalidArgument("p.dims(0) != seq_length: ",
                                        p_tensor->dim_size(0), " vs. ",
                                        seq_length));
    OP_REQUIRES(ctx, p_tensor->dim_size(1) == insize,
                errors::InvalidArgument("p.dims(1) != insize: ",
                                        p_tensor->dim_size(1), " vs. ",
                                        insize));
    OP_REQUIRES(ctx, p_tensor->dim_size(2) == 1,
                errors::InvalidArgument(
                    "p.dims(2) != 1: ", p_tensor->dim_size(2),
                    " vs. ", 1));

    // check the shape of h
    const Tensor* h_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h", &h_tensor));
    OP_REQUIRES(ctx, h_tensor->dims() == 3,
                errors::InvalidArgument("h must be 3D"));
    OP_REQUIRES(ctx, h_tensor->dim_size(0) == seq_length,
                errors::InvalidArgument("h.dims(0) != seq_length: ",
                                        h_tensor->dim_size(0), " vs. ",
                                        seq_length));
    OP_REQUIRES(ctx, h_tensor->dim_size(1) == hidsize,
                errors::InvalidArgument("h.dims(1) != hidsize: ",
                                        h_tensor->dim_size(1), " vs. ",
                                        hidsize));
    OP_REQUIRES(ctx, h_tensor->dim_size(2) == 1,
                errors::InvalidArgument(
                    "h.dims(2) != 1: ", h_tensor->dim_size(2),
                    " vs. ", 1));

    // w_hh_tensor
    const Tensor* w_hh_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_hh", &w_hh_tensor));
    OP_REQUIRES(ctx, w_hh_tensor->dims() == 2,
                errors::InvalidArgument("w_hh must be 2D"));
    OP_REQUIRES(ctx, w_hh_tensor->dim_size(0) == hidsize,
                errors::InvalidArgument("w_hh.dims(0) != hidsize: ",
                                        w_hh_tensor->dim_size(0), " vs. ",
                                        hidsize));
    OP_REQUIRES(ctx, w_hh_tensor->dim_size(1) == hidsize,
                errors::InvalidArgument(
                    "w_hh.dims(1) != hidsize: ", w_hh_tensor->dim_size(1),
                    " vs. ", hidsize));

    // w_hy_tensor
    const Tensor* w_hy_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_hy", &w_hy_tensor));
    OP_REQUIRES(ctx, w_hy_tensor->dims() == 2,
                errors::InvalidArgument("w_hy must be 2D"));
    OP_REQUIRES(ctx, w_hy_tensor->dim_size(0) == insize,
                errors::InvalidArgument("w_hy.dims(0) != insize: ",
                                        w_hy_tensor->dim_size(0), " vs. ",
                                        insize));
    OP_REQUIRES(ctx, w_hy_tensor->dim_size(1) == hidsize,
                errors::InvalidArgument(
                    "w_hy.dims(1) != hidsize: ", w_hy_tensor->dim_size(1),
                    " vs. ", hidsize));

    // check the shape of h_prev
    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));
    OP_REQUIRES(ctx, h_prev_tensor->dims() == 2,
                errors::InvalidArgument("h_prev must be 2D"));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == hidsize,
                errors::InvalidArgument("h_prev.dims(0) != hidsize: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        hidsize));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == 1,
                errors::InvalidArgument(
                    "h_prev.dims(1) != 1: ", h_prev_tensor->dim_size(1),
                    " vs. ", 1));

    // set shape of outputs
    TensorShape d_w_xh_shape({hidsize, insize});
    Tensor* d_w_xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("d_w_xh", d_w_xh_shape, &d_w_xh_tensor));

    TensorShape d_w_hh_shape({hidsize, hidsize});
    Tensor* d_w_hh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("d_w_hh", d_w_hh_shape, &d_w_hh_tensor));

    TensorShape d_w_hy_shape({insize, hidsize});
    Tensor* d_w_hy_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("d_w_hy", d_w_hy_shape, &d_w_hy_tensor));

    TensorShape d_b_h_shape({hidsize, 1});
    Tensor* d_b_h_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("d_b_h", d_b_h_shape, &d_b_h_tensor));

    TensorShape d_b_y_shape({insize, 1});
    Tensor* d_b_y_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("d_b_y", d_b_y_shape, &d_b_y_tensor));

    Tensor dh_next;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({hidsize, 1}), &dh_next));

    const Device& device = ctx->eigen_device<Device>();

    functor::TensorZero<Device, T>()(device, d_w_xh_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, d_w_hh_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, d_w_hy_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, d_b_h_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, d_b_y_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, dh_next.flat<T>());

    // 
    SliceHelper<Device, T> slicer(ctx);

    for (int64 t = seq_length - 1; t >= 0; --t) {
      const Tensor x_t_tensor = slicer.InputSlice(*x_tensor, t, "x");
      const Tensor p_t_tensor = slicer.InputSlice(*p_tensor, t, "p");
      const Tensor h_t_tensor = slicer.InputSlice(*h_tensor, t, "h");
      const Tensor h_prev_t_tensor = slicer.InputSlice(*h_tensor, t == 0 ? t : t - 1, "h");

      typename TTypes<T>::ConstScalar y_t(&(y_tensor->vec<T>()(t)), 0);

      const Tensor *h_prev_p = h_prev_tensor;
      if(t != 0) {
        h_prev_p = &h_prev_t_tensor;
      }
      
      functor::VanillaRNNCellBprop<Device, T, USE_CUBLAS>(seq_length, insize, hidsize)(
          ctx, device, t,
          x_t_tensor.matrix<T>(), y_t, 
          p_t_tensor.matrix<T>(), h_t_tensor.matrix<T>(), 
          w_hh_tensor->matrix<T>(), w_hy_tensor->matrix<T>(), h_prev_p->matrix<T>(), dh_next.matrix<T>(), 
          d_w_xh_tensor->matrix<T>(), d_w_hh_tensor->matrix<T>(), d_w_hy_tensor->matrix<T>(),
          d_b_h_tensor->matrix<T>(), d_b_y_tensor->matrix<T>());

      slicer.FinishTimeStep();
    }

  }

 private:
  int32 hidsize;
};

#define REGISTER_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("VanillaRNNGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      VanillaRNNGradOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void TensorCopy<GPUDevice, T>::operator()(const GPUDevice& d,                \
                                            typename TTypes<T>::ConstFlat src, \
                                            typename TTypes<T>::Flat dst);     \
                                                                               \
  template <>                                                                  \
  void TensorCopyUnaligned<GPUDevice, T>::operator()(                          \
      const GPUDevice& d, typename TTypes<T>::UnalignedConstFlat src,          \
      typename TTypes<T>::Flat dst);                                           \
                                                                               \
  template <>                                                                  \
  void TensorCopyToUnaligned<GPUDevice, T>::operator()(                        \
      const GPUDevice& d, typename TTypes<T>::ConstFlat src,                   \
      typename TTypes<T>::UnalignedFlat dst);                                  \
                                                                               \
  template <>                                                                  \
  void TensorAdd<GPUDevice, T>::operator()(                                    \
      const GPUDevice& d, typename TTypes<T>::ConstFlat a,                     \
      typename TTypes<T>::ConstFlat b, typename TTypes<T>::Flat c);            \
                                                                               \
  template <>                                                            \
  void TensorZero<GPUDevice, T>::operator()(const GPUDevice& d,          \
                                            typename TTypes<T>::Flat t); \
                                                                         \
  extern template struct TensorZero<GPUDevice, T>;                       \
                                                                         \
  template <>                                                            \
  void TensorUnalignedZero<GPUDevice, T>::operator()(                    \
      const GPUDevice& d, typename TTypes<T>::UnalignedFlat t);          \
                                                                         \
  extern template struct TensorUnalignedZero<GPUDevice, T>;               \
                                                                          \
  template <>                                                                  \
  void VanillaRNNCellBprop<GPUDevice, T, true>::operator()(                         \
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
      typename TTypes<T>::Matrix d_b_y_out);                                      \
                                                                               \
  extern template struct VanillaRNNCellBprop<GPUDevice, T, true>;

DECLARE_GPU_SPEC(float);
// DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

#define REGISTER_GPU_KERNEL(T)                           \
  REGISTER_KERNEL_BUILDER(Name("VanillaRNNGrad")         \
                              .Device(DEVICE_GPU)        \
                              .HostMemory("y")           \
                              .TypeConstraint<T>("T"),   \
                          VanillaRNNGradOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
// REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
