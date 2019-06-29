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
// to compute loss, dw_y, db_y values.
// Ref: sparse_xent_op and lstm_ops modules
//
// Author: Rock Zhuang
// Date  : May 31, 2019
// 

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/rnn_softmaxloss_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"

// #define VERBOSE 1
// #define TESTING 1

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Index>
Status CheckInvalidLabelIndex(const Tensor& labels, int64 max_index) {
  if (labels.NumElements() == 0) return Status::OK();
  const auto label_values = labels.vec<Index>();
  int64 bad_index;
  auto min_max_dim_value = std::minmax_element(
      label_values.data(), label_values.data() + label_values.size());
  if (*min_max_dim_value.first < 0 || *min_max_dim_value.second >= max_index) {
    bad_index = (*min_max_dim_value.first < 0) ? *min_max_dim_value.first
                                               : *min_max_dim_value.second;
    return errors::InvalidArgument(
        "Received a label value of ", bad_index,
        " which is outside the valid range of [0, ", max_index,
        ").  Label values: ", labels.SummarizeValue(labels.NumElements()));
  }
  return Status::OK();
}


namespace {
// SliceHelper is copied from lstm_ops modules

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

  const Tensor InputSliceFromTwoDims(const Tensor& t, int pos, const string& name) {
    Tensor res = UnalignedSliceFromTwoDims(t, pos);
    if (res.IsAligned()) {
      return res;
    } else {
      return AlignTensor(res, name);
    }
  }


  const Tensor InputSliceKeepTwoDim(const Tensor& t, int pos, const string& name) {
    Tensor res = UnalignedSliceKeepTwoDim(t, pos);
    if (res.IsAligned()) {
      return res;
    } else {
      return AlignTensor(res, name);
    }
  }

  const Tensor InputSliceKeepOneDim(const Tensor& t, int pos, const string& name) {
    Tensor res = UnalignedSliceKeepOneDim(t, pos);
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

  Tensor OutputSliceFromTwoDims(Tensor* t, int pos, const string& name) {
    Tensor res = UnalignedSliceFromTwoDims(*t, pos);
    if (res.IsAligned()) {
      return res;
    } else {
      Tensor aligned = AlignTensor(res, name);
      copy_out_.emplace_back(res, aligned);
      return aligned;
    }
  }

  Tensor OutputSliceKeepTwoDim(Tensor* t, int pos, const string& name) {
    Tensor res = UnalignedSliceKeepTwoDim(*t, pos);
    if (res.IsAligned()) {
      return res;
    } else {
      Tensor aligned = AlignTensor(res, name);
      copy_out_.emplace_back(res, aligned);
      return aligned;
    }
  }

  Tensor OutputSliceKeepOneDim(Tensor* t, int pos, const string& name) {
    Tensor res = UnalignedSliceKeepOneDim(*t, pos);
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

  Tensor UnalignedSliceFromTwoDims(const Tensor& t, int pos) const {
    Tensor res;
    // CHECK should never fail here, since the number of elements must match
    CHECK(res.CopyFrom(t.Slice(pos, pos + 1), {t.dim_size(1)}));
    return res;
  }

  Tensor UnalignedSliceKeepTwoDim(const Tensor& t, int pos) const {
    Tensor res;
    // CHECK should never fail here, since the number of elements must match
    CHECK(res.CopyFrom(t.Slice(pos, pos + 1), {1, t.dim_size(1)}));
    return res;
  }

  Tensor UnalignedSliceKeepOneDim(const Tensor& t, int pos) const {
    Tensor res;
    // CHECK should never fail here, since the number of elements must match
    CHECK(res.CopyFrom(t.Slice(pos, pos + 1), {1}));
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
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(int64);
#undef DECLARE_GPU_SPEC
}  // end namespace functor


template <typename Device, typename T, typename Index>
class RNNSoftmaxLossOp : public OpKernel {
 public:
  explicit RNNSoftmaxLossOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& h_tensor = context->input(0);
    const Tensor& labels = context->input(1);
    const Tensor& w_y_tensor = context->input(2);
    const Tensor& b_y_tensor = context->input(3);

    // LOG(INFO) << __FUNCTION__ << "----------------------------b_y_tensor:" << std::endl << b_y_tensor.DebugString();

    int time_len = h_tensor.dim_size(0);
    int batch_size = h_tensor.dim_size(1);
    int num_units = h_tensor.dim_size(2);
    int input_size = w_y_tensor.dim_size(0);

    OP_REQUIRES(context, h_tensor.dims() == 3,
              errors::InvalidArgument("h_tensor must be 3-dimensional: ",
                                      h_tensor.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(labels.shape()),
                errors::InvalidArgument("labels must be 2-D, but got shape ",
                                        labels.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(w_y_tensor.shape()),
                errors::InvalidArgument("w_y_tensor must be 2-D, but got shape ",
                                        w_y_tensor.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(b_y_tensor.shape()),
                errors::InvalidArgument("labels must be 1-D, but got shape ",
                                        b_y_tensor.shape().DebugString()));

    Tensor scratch; // temp variable
    TensorShape scratch_shape({batch_size});
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   scratch_shape, &scratch));

    Tensor backprop; // dy
    TensorShape backprop_shape({batch_size, input_size});
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   backprop_shape, &backprop));

    Tensor* loss_out = nullptr; // loss
    OP_REQUIRES_OK(context, context->allocate_output("loss", labels.shape(), &loss_out));

    Tensor* p_out = nullptr; // p
    TensorShape p_shape({time_len, batch_size, input_size});
    OP_REQUIRES_OK(context, context->allocate_output("p", p_shape, &p_out));

    Tensor *logits_out; // y_s or y'
    TensorShape logits_shape({time_len, batch_size, input_size});
    OP_REQUIRES_OK(context, context->allocate_output("logits", logits_shape, &logits_out));

    const Device& device = context->eigen_device<Device>();

    functor::TensorZero<Device, T>()(device, loss_out->flat<T>());
    functor::TensorZero<Device, T>()(device, p_out->flat<T>());
    functor::TensorZero<Device, T>()(device, logits_out->flat<T>());

    SliceHelper<Device, T> slicer(context);
    SliceHelper<Device, Index> slicer2(context);

    // Reverse order
    for(int t = time_len - 1; t >= 0; t--) {  
      const Tensor h_sub_tensor = slicer.InputSlice(h_tensor, t, "h_sub");
      const Tensor labels_sub_tensor = slicer2.InputSliceFromTwoDims(labels, t, "labels_sub");
  
      Tensor loss_out_tensor = slicer.OutputSliceFromTwoDims(loss_out, t, "loss_sub");
      Tensor p_out_tensor = slicer.OutputSlice(p_out, t, "p_sub");
      Tensor logits = slicer.OutputSlice(logits_out, t, "logits_sub");

      functor::SparseXentFunctor<Device, T, Index> functor;
      functor(context, device, h_sub_tensor.matrix<T>(),
              labels_sub_tensor.vec<Index>(), w_y_tensor.matrix<T>(), b_y_tensor.vec<T>(), 
              logits.matrix<T>(), scratch.vec<T>(), backprop.matrix<T>(),
              p_out_tensor.matrix<T>(), loss_out_tensor.vec<T>());

      slicer.FinishTimeStep();
      slicer2.FinishTimeStep();
    }
  }
};

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
namespace functor {
template <typename T, typename Index>
struct SparseXentFunctor<CPUDevice, T, Index> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d, typename TTypes<T>::ConstMatrix h,
                  typename TTypes<Index>::ConstVec labels,
                  typename TTypes<T>::ConstMatrix w_y, typename TTypes<T>::ConstVec b_y, 
                  typename TTypes<T>::Matrix logits, typename TTypes<T>::Vec scratch, typename TTypes<T>::Matrix backprop, 
                  typename TTypes<T>::Matrix p, typename TTypes<T>::Vec loss) {
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

    TensorBlasGemm<CPUDevice, T, false>::compute(
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

    Eigen::array<Eigen::DenseIndex, 2> b_shape({batch_size, 1});
    Eigen::array<Eigen::DenseIndex, 2> bcast({1, num_classes});
    p.device(d) = backprop.exp() / scratch.reshape(b_shape).broadcast(bcast);

    // loss:
    //  sum(-labels *
    //     ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
    //  along classes
    generator::SparseXentLossGenerator<T, Index> sparse_xent_loss_gen(
        sparse_xent_helpers::To32BitConst<T>(backprop),
        sparse_xent_helpers::To32BitConst<T>(scratch), To32Bit(labels),
        backprop.dimension(1) /* max_depth */);
    To32Bit(loss).device(d) =
        To32Bit(backprop).generate(sparse_xent_loss_gen).sum(along_class);
  }
};

}  // namespace functor

#define REGISTER(Dev, T, Index)                   \
  REGISTER_KERNEL_BUILDER(                        \
      Name("RNNSoftmaxLoss") \
          .Device(DEVICE_##Dev)                   \
          .TypeConstraint<T>("T")                 \
          .TypeConstraint<Index>("Tlabels"),      \
      RNNSoftmaxLossOp<Dev##Device, T, Index>);
REGISTER(CPU, float, int32)
REGISTER(CPU, float, int64)
REGISTER(CPU, double, int32)
REGISTER(CPU, double, int64)
REGISTER(CPU, Eigen::half, int32)
REGISTER(CPU, Eigen::half, int64)

#if GOOGLE_CUDA
REGISTER(GPU, float, int32)
REGISTER(GPU, float, int64)
REGISTER(GPU, Eigen::half, int32)
REGISTER(GPU, Eigen::half, int64)

namespace functor {
#define DECLARE_GPU_SPEC(T)                                              \
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
  extern template struct TensorCopy<GPUDevice, T>;                             \
  extern template struct TensorAdd<GPUDevice, T>;                              

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(int64);
DECLARE_GPU_SPEC(int32);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

#endif  // GOOGLE_CUDA

#undef REGISTER

//
// Grad
//

template <typename Device, typename T, typename Index>
class RNNSoftmaxLossGradOp : public OpKernel {
 public:
  explicit RNNSoftmaxLossGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& h_tensor = context->input(0);
    const Tensor& labels = context->input(1);
    const Tensor& w_y_tensor = context->input(2);
    const Tensor& b_y_tensor = context->input(3);
    const Tensor& p_tensor = context->input(4);

    // LOG(INFO) << __FUNCTION__ << "----------------------------p_tensor:" << std::endl << p_tensor.DebugString();

    int time_len = h_tensor.dim_size(0);
    int batch_size = h_tensor.dim_size(1);
    int num_units = h_tensor.dim_size(2);
    int input_size = w_y_tensor.dim_size(0);

    OP_REQUIRES(context, h_tensor.dims() == 3,
              errors::InvalidArgument("h_tensor must be 3-dimensional: ",
                                      h_tensor.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(labels.shape()),
                errors::InvalidArgument("labels must be 2-D, but got shape ",
                                        labels.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(w_y_tensor.shape()),
                errors::InvalidArgument("w_y_tensor must be 2-D, but got shape ",
                                        w_y_tensor.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(b_y_tensor.shape()),
                errors::InvalidArgument("labels must be 1-D, but got shape ",
                                        b_y_tensor.shape().DebugString()));
    OP_REQUIRES(context, p_tensor.dims() == 3,
              errors::InvalidArgument("p_tensor must be 3-dimensional: ",
                                      p_tensor.shape().DebugString()));

    Tensor backprop; // dy
    TensorShape backprop_shape({batch_size, input_size});
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   backprop_shape, &backprop));

    // output 
    Tensor* h_grad = nullptr; // dh
    OP_REQUIRES_OK(context, context->allocate_output("h_grad", h_tensor.shape(), &h_grad));

    // {TIME_LEN, BATCH_SIZE, NUM_UNIT}
    Tensor* cs_grad = nullptr; // cs
    TensorShape cs_grad_shape({time_len, batch_size, num_units});
    OP_REQUIRES_OK(context, context->allocate_output("cs_grad", cs_grad_shape, &cs_grad));

    TensorShape dw_y_shape({input_size, num_units});
    Tensor* dw_y_tensor;
    OP_REQUIRES_OK(context, context->allocate_output("dw_y", dw_y_shape, &dw_y_tensor));

    TensorShape db_y_shape({input_size});
    Tensor* db_y_tensor;
    OP_REQUIRES_OK(context, context->allocate_output("db_y", db_y_shape, &db_y_tensor));


    const Device& device = context->eigen_device<Device>();

    functor::TensorZero<Device, T>()(device, h_grad->flat<T>());
    functor::TensorZero<Device, T>()(device, cs_grad->flat<T>()); // TODO: how to do with cs_grad? 
    functor::TensorZero<Device, T>()(device, dw_y_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, db_y_tensor->flat<T>());

    SliceHelper<Device, T> slicer(context);
    SliceHelper<Device, Index> slicer2(context);

    // Reverse order
    for(int t = time_len - 1; t >= 0; t--) {  
      const Tensor h_sub_tensor = slicer.InputSlice(h_tensor, t, "h_sub");
      const Tensor labels_sub_tensor = slicer2.InputSliceFromTwoDims(labels, t, "labels_sub");
  
      Tensor p_sub_tensor = slicer.InputSlice(p_tensor, t, "p_sub");

      Tensor hgrad_tensor = slicer.OutputSlice(h_grad, t, "h_grad_sub");

      // LOG(INFO) << __FUNCTION__ << "----------------------------p_sub_tensor:" << std::endl << p_sub_tensor.DebugString();

      functor::GradFunctor<Device, T, Index> functor;
      functor(context, device, h_sub_tensor.matrix<T>(),
              labels_sub_tensor.vec<Index>(), p_sub_tensor.matrix<T>(),
              w_y_tensor.matrix<T>(), b_y_tensor.vec<T>(), 
              backprop.matrix<T>(), 
              hgrad_tensor.matrix<T>(), dw_y_tensor->matrix<T>(), db_y_tensor->vec<T>());

      slicer.FinishTimeStep();
      slicer2.FinishTimeStep();
    }
  }
};



// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
namespace functor {
template <typename T, typename Index>
struct GradFunctor<CPUDevice, T, Index> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d, typename TTypes<T>::ConstMatrix h,
                  typename TTypes<Index>::ConstVec labels, typename TTypes<T>::Matrix p, 
                  typename TTypes<T>::ConstMatrix w_y, typename TTypes<T>::ConstVec b_y, 
                  typename TTypes<T>::Matrix backprop, 
                  typename TTypes<T>::Matrix h_grad,
                  typename TTypes<T>::Matrix dw_y, typename TTypes<T>::Vec db_y) {
    // backprop (dy here): prob - labels
    generator::RNNSoftmaxLossGradGenerator<T, Index> sparse_xent_grad_gen(
        sparse_xent_helpers::To32BitConst<T>(p), To32Bit(labels),
        p.dimension(1) /* max_depth */);
    To32Bit(backprop).device(d) =
        To32Bit(p).generate(sparse_xent_grad_gen);

    // dW_y += np.dot(dy, h.T), for a batch
    typename TTypes<T>::ConstMatrix const_backprop(backprop.data(), backprop.dimensions());
    TensorBlasGemm<CPUDevice, T, false>::compute(
        ctx, d, true, false, 1.f, const_backprop, h, 1.f, dw_y);

    // db_y += dy, for a batch
    db_y.device(d) += backprop.sum(Eigen::array<int, 1>({0}));

    // dh = np.dot(W_y.T, dy), for a batch
    TensorBlasGemm<CPUDevice, T, false>::compute(
        ctx, d, false, false, 1.f, const_backprop, w_y, 0.f, h_grad);
  }
};

}  // namespace functor


#define REGISTER(Dev, T, Index)                   \
  REGISTER_KERNEL_BUILDER(                        \
      Name("RNNSoftmaxLossGrad") \
          .Device(DEVICE_##Dev)                   \
          .TypeConstraint<T>("T")                 \
          .TypeConstraint<Index>("Tlabels"),      \
      RNNSoftmaxLossGradOp<Dev##Device, T, Index>);
REGISTER(CPU, float, int32)
REGISTER(CPU, float, int64)
REGISTER(CPU, double, int32)
REGISTER(CPU, double, int64)
REGISTER(CPU, Eigen::half, int32)
REGISTER(CPU, Eigen::half, int64)

#if GOOGLE_CUDA
REGISTER(GPU, float, int32)
REGISTER(GPU, float, int64)
REGISTER(GPU, Eigen::half, int32)
REGISTER(GPU, Eigen::half, int64)
#endif

#undef REGISTER

}  // namespace tensorflow
