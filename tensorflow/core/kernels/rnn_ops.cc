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
// rnn ops, Vanalli RNN for now
// Author: Rock Zhuang
// Date  : Dec 20, 2018
// 

//
// Ref: lstm_ops.cc
// Eigen Tensor Ref: https://github.com/eigenteam/eigen-git-mirror/tree/master/unsupported/Eigen/CXX11/src/Tensor
//   and https://eigen.tuxfamily.org/dox-devel/unsupported/eigen_tensors.html#title0
// 

#if GOOGLE_CUDA_TODO // GPU not supported yet
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA_TODO

#define VERBOSE 1
#define TESTING 1

#define EIGEN_USE_THREADS

#include <memory>
#include <vector>
#include <algorithm>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/rnn_ops.h"

namespace tensorflow {

// typedef Eigen::ThreadPoolDevice CPUDevice;
// typedef Eigen::GpuDevice GPUDevice;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
void VanillaRNNCellFpropWithEigen(
    const VanillaRNNCell& cell, OpKernelContext* ctx, const CPUDevice& d, const int64 t,
    typename TTypes<T>::ConstMatrix x,
    typename TTypes<T>::ConstMatrix y,
    typename TTypes<T>::ConstMatrix h_prev,
    typename TTypes<T>::ConstMatrix w_xh,
    typename TTypes<T>::ConstMatrix w_hh,
    typename TTypes<T>::ConstMatrix w_hy,
    typename TTypes<T>::ConstMatrix b_h,
    typename TTypes<T>::ConstMatrix b_y,
    typename TTypes<T>::Matrix p_out, typename TTypes<T>::Matrix h_out, typename TTypes<T>::Scalar loss_out) {

#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "---------------------------------------------------------sequence number:" << std::endl << t;
  LOG(INFO) << __FUNCTION__ << "----------------------------x:" << std::endl << x;
  LOG(INFO) << __FUNCTION__ << "----------------------------y:" << std::endl << y;
#endif
  int y_index = 0;
  int insize = y.dimension(0);
  for(; y_index < insize; y_index++) {
    T scalar = y(y_index, 0);
    if(scalar > 0.0f)
      break;
  }

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
  Eigen::array<Eigen::DenseIndex, 2> bcast({insize, 1});
  auto p = y_c_exp / y_c_exp.sum().reshape(b_shape).broadcast(bcast);

#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "----------------------------softmax h:" << std::endl << h<< std::endl;
  LOG(INFO) << __FUNCTION__ << "----------------------------softmax y_c:" << std::endl << y_c<< std::endl;
  LOG(INFO) << __FUNCTION__ << "----------------------------softmax y_c_exp:" << std::endl << y_c_exp<< std::endl;
  LOG(INFO) << __FUNCTION__ << "----------------------------softmax p:" << std::endl << p;
#endif

  // Corresponding Python code:
  //  loss += -np.log(p[t][y[t],0])#softmax (cross-entropy loss)
  // 
  auto py = p.chip(y_index, 0);
  loss_out.device(d) = -py.log();

  p_out.device(d) = p;
  h_out.device(d) = h;

#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "----------------------------py:" << std::endl << py;
  LOG(INFO) << __FUNCTION__ << "----------------------------loss:" << std::endl << loss_out;
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

    auto c = a * b;
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

#define DEFINE_CPU_SPECS(T)                                                   \
  template <>                                                                 \
  void VanillaRNNCellFprop<CPUDevice, T, false /* USE_CUBLAS */>::operator()(  \
      OpKernelContext* ctx, const CPUDevice& d, const int64 t,                      \
      typename TTypes<T>::ConstMatrix x,                                      \
      typename TTypes<T>::ConstMatrix y,                                      \
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
  template struct VanillaRNNCellFprop<CPUDevice, T, false /* USE_CUBLAS */>;

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
    OP_REQUIRES(ctx, y_tensor->dims() == 2, errors::InvalidArgument("y must be 2D"));

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
    TensorShape p_cell_shape({insize, 1});
    Tensor* p_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("p", p_cell_shape, &p_tensor));

    TensorShape hid_shape({hidsize, 1});
    Tensor* h_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("h", hid_shape, &h_tensor));

    TensorShape loss_shape({});
    Tensor* loss_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", loss_shape, &loss_tensor));

    // 
    const Device& device = ctx->eigen_device<Device>();
    SliceHelper<Device, T> slicer(ctx);

    *h_tensor = *h_prev_tensor;

    for (int64 t = 0; t < seq_length; ++t) {
      const Tensor x_t_tensor = slicer.InputSlice(*x_tensor, t, "x");
      
      const Tensor *tmp = h_tensor;
      Tensor loss(DT_FLOAT, TensorShape({}));

      functor::VanillaRNNCellFprop<Device, T, USE_CUBLAS>(seq_length, insize)(
          ctx, device, t,
          x_t_tensor.matrix<T>(), y_tensor->matrix<T>(), tmp->matrix<T>(), 
          w_xh_tensor->matrix<T>(), w_hh_tensor->matrix<T>(), w_hy_tensor->matrix<T>(),
          b_h_tensor->matrix<T>(), b_y_tensor->matrix<T>(), 
          p_tensor->matrix<T>(), h_tensor->matrix<T>(), loss.scalar<T>());

      // cumulate
      loss_tensor->scalar<T>()() += loss.scalar<T>()();

#ifdef VERBOSE
  LOG(INFO) << __FUNCTION__ << "----------------------------loss:" << std::endl << loss.scalar<T>()() << ", " << loss_tensor->scalar<T>()();
#endif
      slicer.FinishTimeStep();
    }

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

#if GOOGLE_CUDA_TODO
namespace functor {
#define DECLARE_GPU_SPEC(T)                                              \
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
  extern template struct TensorUnalignedZero<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
// DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

#define REGISTER_GPU_KERNEL(T)                           \
  REGISTER_KERNEL_BUILDER(Name("VanillaRNN")              \
                              .Device(DEVICE_GPU)        \
                              .HostMemory("seq_len_max") \
                              .TypeConstraint<T>("T"),   \
                          VanillaRNNOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
// REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA_TODO

template <typename Device, typename T, bool USE_CUBLAS>
class VanillaRNNGradOp : public OpKernel {
 public:
  explicit VanillaRNNGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* seq_len_max_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("seq_len_max", &seq_len_max_tensor));

    const Tensor* x;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x));
    OP_REQUIRES(ctx, x->dims() == 3, errors::InvalidArgument("x must be 3D"));
    const int64 timelen = x->dim_size(0);
    const int64 batch_size = x->dim_size(1);
    const int64 input_size = x->dim_size(2);

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));
    const int64 cell_size = w_tensor->dim_size(1) / 4;
    OP_REQUIRES(ctx, input_size + cell_size == w_tensor->dim_size(0),
                errors::InvalidArgument(
                    "w matrix rows don't match: ", input_size + cell_size,
                    " vs. ", w_tensor->dim_size(0)));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));
    OP_REQUIRES(
        ctx, cell_size == b_tensor->dim_size(0) / 4,
        errors::InvalidArgument("w and b cell_size don't match: ", cell_size,
                                " vs. ", b_tensor->dim_size(0)));

    const Tensor* i_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("i", &i_out));

    const Tensor* cs_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs", &cs_out));

    const Tensor* f_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("f", &f_out));

    const Tensor* o_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("o", &o_out));

    const Tensor* ci_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ci", &ci_out));

    const Tensor* co_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("co", &co_out));

    const Tensor* h_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h", &h_out));

    const Tensor* cs_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_grad", &cs_grad));

    const Tensor* h_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_grad", &h_grad));

    TensorShape batch_input_shape({timelen, batch_size, input_size});
    Tensor* x_grad;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("x_grad", batch_input_shape, &x_grad));

    Tensor* cs_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("cs_prev_grad", cs_prev_tensor->shape(),
                                        &cs_prev_grad_tensor));

    Tensor* h_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("h_prev_grad", h_prev_tensor->shape(),
                                        &h_prev_grad_tensor));

    Tensor* w_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("w_grad", w_tensor->shape(), &w_grad_tensor));

    Tensor* wci_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("wci_grad", wci_tensor->shape(),
                                             &wci_grad_tensor));

    Tensor* wcf_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("wcf_grad", wcf_tensor->shape(),
                                             &wcf_grad_tensor));

    Tensor* wco_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("wco_grad", wco_tensor->shape(),
                                             &wco_grad_tensor));

    Tensor* b_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("b_grad", b_tensor->shape(), &b_grad_tensor));

    TensorShape batch_cell_shape({batch_size, cell_size});

    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::v(),
                            TensorShape({batch_size, input_size + cell_size}),
                            &xh_tensor));

    Tensor xh_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           xh_tensor.shape(), &xh_grad_tensor));

    Tensor do_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &do_tensor));

    Tensor dcs_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &dcs_tensor));

    Tensor dci_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &dci_tensor));

    Tensor df_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &df_tensor));

    Tensor di_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &di_tensor));

    Tensor dicfo_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                      TensorShape({batch_size, cell_size * 4}),
                                      &dicfo_tensor));

    Tensor cs_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &cs_grad_tensor));

    Tensor h_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &h_grad_tensor));

    const Device& device = ctx->eigen_device<Device>();

    functor::TensorZero<Device, T>()(device, cs_grad_tensor.flat<T>());
    functor::TensorZero<Device, T>()(device, cs_prev_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, h_grad_tensor.flat<T>());
    functor::TensorZero<Device, T>()(device, h_prev_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, w_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, wci_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, wcf_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, wco_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, b_grad_tensor->flat<T>());

    const int64 seq_len_max = seq_len_max_tensor->scalar<int64>()();
    for (int64 t = seq_len_max - 1; t >= 0; --t) {
      
    //   functor::VanillaRNNBprop<Device, T, USE_CUBLAS>(batch_size, input_size,
    //                                                  cell_size)(
    //       ctx, device, use_peephole_, x_tensor.matrix<T>(),
    //       cs_prev_tensor2.matrix<T>(), h_prev_tensor2.matrix<T>(),
    //       w_tensor->matrix<T>(), wci_tensor->vec<T>(), wcf_tensor->vec<T>(),
    //       wco_tensor->vec<T>(), b_tensor->vec<T>(), xh_tensor.matrix<T>(),
    //       i_tensor.matrix<T>(), cs_tensor.matrix<T>(), f_tensor.matrix<T>(),
    //       o_tensor.matrix<T>(), ci_tensor.matrix<T>(), co_tensor.matrix<T>(),
    //       const_cs_grad_tensor.matrix<T>(), const_h_grad_tensor.matrix<T>(),
    //       do_tensor.matrix<T>(), dcs_tensor.matrix<T>(), dci_tensor.matrix<T>(),
    //       df_tensor.matrix<T>(), di_tensor.matrix<T>(),
    //       dicfo_tensor.matrix<T>(), cs_prev_grad_tensor->matrix<T>(),
    //       h_prev_grad_tensor->matrix<T>(), xh_grad_tensor.matrix<T>(),
    //       x_grad_tensor.matrix<T>(), w_grad_tensor->matrix<T>(),
    //       wci_grad_tensor->vec<T>(), wcf_grad_tensor->vec<T>(),
    //       wco_grad_tensor->vec<T>(), b_grad_tensor->vec<T>());
    }

  }

 private:
  bool use_peephole_;
};

#define REGISTER_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("VanillaRNNGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      VanillaRNNGradOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA_TODO
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
  template <>                                                                  \
  void VanillaRNNBprop<GPUDevice, T, true>::operator()(                         \
      OpKernelContext* ctx, const GPUDevice& d, bool use_peephole,             \
      typename TTypes<T>::ConstMatrix x,                                       \
      typename TTypes<T>::ConstMatrix cs_prev,                                 \
      typename TTypes<T>::ConstMatrix h_prev,                                  \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci,     \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,      \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::Matrix xh,           \
      typename TTypes<T>::ConstMatrix i, typename TTypes<T>::ConstMatrix cs,   \
      typename TTypes<T>::ConstMatrix f, typename TTypes<T>::ConstMatrix o,    \
      typename TTypes<T>::ConstMatrix ci, typename TTypes<T>::ConstMatrix co,  \
      typename TTypes<T>::ConstMatrix cs_grad,                                 \
      typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_,  \
      typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,          \
      typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,            \
      typename TTypes<T>::Matrix dicfo,                                        \
      typename TTypes<T>::Matrix cs_prev_grad,                                 \
      typename TTypes<T>::Matrix h_prev_grad,                                  \
      typename TTypes<T>::Matrix xh_grad, typename TTypes<T>::Matrix x_grad,   \
      typename TTypes<T>::Matrix w_grad, typename TTypes<T>::Vec wci_grad,     \
      typename TTypes<T>::Vec wcf_grad, typename TTypes<T>::Vec wco_grad,      \
      typename TTypes<T>::Vec b_grad);                                         \
                                                                               \
  extern template struct TensorCopy<GPUDevice, T>;                             \
  extern template struct TensorAdd<GPUDevice, T>;                              \
  extern template struct VanillaRNNBprop<GPUDevice, T, true>;

DECLARE_GPU_SPEC(float);
// DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

#define REGISTER_GPU_KERNEL(T)                           \
  REGISTER_KERNEL_BUILDER(Name("VanillaRNNGrad")          \
                              .Device(DEVICE_GPU)        \
                              .HostMemory("seq_len_max") \
                              .TypeConstraint<T>("T"),   \
                          VanillaRNNGradOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
// REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA_TODO

}  // end namespace tensorflow
