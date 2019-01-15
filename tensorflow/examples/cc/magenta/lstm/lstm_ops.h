// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_LSTM_OPS_H_
#define TENSORFLOW_CC_OPS_LSTM_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup lstm_ops Lstm Ops
/// @{

/// Computes the LSTM cell forward propagation for all the time steps.
///
/// This is equivalent to applying LSTMBlockCell in a loop, like so:
///
/// ```python
/// for x1 in unpack(x):
///   i1, cs1, f1, o1, ci1, co1, h1 = LSTMBlock(
///     x1, cs_prev, h_prev, w, wci, wcf, wco, b)
///   cs_prev = cs1
///   h_prev = h1
///   i.append(i1)
///   cs.append(cs1)
///   f.append(f1)
///   o.append(o1)
///   ci.append(ci1)
///   co.append(co1)
///   h.append(h1)
/// return pack(i), pack(cs), pack(f), pack(o), pack(ci), pack(ch), pack(h)
/// ```
///
/// Arguments:
/// * scope: A Scope object
/// * seq_len_max: Maximum time length actually used by this input. Outputs are padded
/// with zeros beyond this length.
/// * x: The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
/// * cs_prev: Value of the initial cell state.
/// * h_prev: Initial output of cell (to be used for peephole).
/// * w: The weight matrix.
/// * wci: The weight matrix for input gate peephole connection.
/// * wcf: The weight matrix for forget gate peephole connection.
/// * wco: The weight matrix for output gate peephole connection.
/// * b: The bias vector.
///
/// Optional attributes (see `Attrs`):
/// * forget_bias: The forget gate bias.
/// * cell_clip: Value to clip the 'cs' value to.
/// * use_peephole: Whether to use peephole weights.
///
/// Returns:
/// * `Output` i: The input gate over the whole time sequence.
/// * `Output` cs: The cell state before the tanh over the whole time sequence.
/// * `Output` f: The forget gate over the whole time sequence.
/// * `Output` o: The output gate over the whole time sequence.
/// * `Output` ci: The cell input over the whole time sequence.
/// * `Output` co: The cell after the tanh over the whole time sequence.
/// * `Output` h: The output h vector over the whole time sequence.
class BlockLSTM {
 public:
  /// Optional attribute setters for BlockLSTM
  struct Attrs {
    /// The forget gate bias.
    ///
    /// Defaults to 1
    TF_MUST_USE_RESULT Attrs ForgetBias(float x) {
      Attrs ret = *this;
      ret.forget_bias_ = x;
      return ret;
    }

    /// Value to clip the 'cs' value to.
    ///
    /// Defaults to 3
    TF_MUST_USE_RESULT Attrs CellClip(float x) {
      Attrs ret = *this;
      ret.cell_clip_ = x;
      return ret;
    }

    /// Whether to use peephole weights.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UsePeephole(bool x) {
      Attrs ret = *this;
      ret.use_peephole_ = x;
      return ret;
    }

    float forget_bias_ = 1.0f;
    float cell_clip_ = 3.0f;
    bool use_peephole_ = false;
  };
  BlockLSTM(const ::tensorflow::Scope& scope, ::tensorflow::Input seq_len_max,
          ::tensorflow::Input x, ::tensorflow::Input cs_prev,
          ::tensorflow::Input h_prev, ::tensorflow::Input w,
          ::tensorflow::Input wci, ::tensorflow::Input wcf, ::tensorflow::Input
          wco, ::tensorflow::Input b);
  BlockLSTM(const ::tensorflow::Scope& scope, ::tensorflow::Input seq_len_max,
          ::tensorflow::Input x, ::tensorflow::Input cs_prev,
          ::tensorflow::Input h_prev, ::tensorflow::Input w,
          ::tensorflow::Input wci, ::tensorflow::Input wcf, ::tensorflow::Input
          wco, ::tensorflow::Input b, const BlockLSTM::Attrs& attrs);

  static Attrs ForgetBias(float x) {
    return Attrs().ForgetBias(x);
  }
  static Attrs CellClip(float x) {
    return Attrs().CellClip(x);
  }
  static Attrs UsePeephole(bool x) {
    return Attrs().UsePeephole(x);
  }

  Operation operation;
  ::tensorflow::Output i;
  ::tensorflow::Output cs;
  ::tensorflow::Output f;
  ::tensorflow::Output o;
  ::tensorflow::Output ci;
  ::tensorflow::Output co;
  ::tensorflow::Output h;
};

/// Computes the LSTM cell backward propagation for the entire time sequence.
///
/// This implementation is to be used in conjunction of LSTMBlock.
///
/// Arguments:
/// * scope: A Scope object
/// * seq_len_max: Maximum time length actually used by this input. Outputs are padded
/// with zeros beyond this length.
/// * x: The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
/// * cs_prev: Value of the initial cell state.
/// * h_prev: Initial output of cell (to be used for peephole).
/// * w: The weight matrix.
/// * wci: The weight matrix for input gate peephole connection.
/// * wcf: The weight matrix for forget gate peephole connection.
/// * wco: The weight matrix for output gate peephole connection.
/// * b: The bias vector.
/// * i: The input gate over the whole time sequence.
/// * cs: The cell state before the tanh over the whole time sequence.
/// * f: The forget gate over the whole time sequence.
/// * o: The output gate over the whole time sequence.
/// * ci: The cell input over the whole time sequence.
/// * co: The cell after the tanh over the whole time sequence.
/// * h: The output h vector over the whole time sequence.
/// * cs_grad: The current gradient of cs.
/// * h_grad: The gradient of h vector.
/// * use_peephole: Whether to use peephole weights.
///
/// Returns:
/// * `Output` x_grad: The gradient of x to be back-propped.
/// * `Output` cs_prev_grad: The gradient of cs_prev to be back-propped.
/// * `Output` h_prev_grad: The gradient of h_prev to be back-propped.
/// * `Output` w_grad: The gradient for w to be back-propped.
/// * `Output` wci_grad: The gradient for wci to be back-propped.
/// * `Output` wcf_grad: The gradient for wcf to be back-propped.
/// * `Output` wco_grad: The gradient for wco to be back-propped.
/// * `Output` b_grad: The gradient for w to be back-propped.
class BlockLSTMGrad {
 public:
  BlockLSTMGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
              seq_len_max, ::tensorflow::Input x, ::tensorflow::Input cs_prev,
              ::tensorflow::Input h_prev, ::tensorflow::Input w,
              ::tensorflow::Input wci, ::tensorflow::Input wcf,
              ::tensorflow::Input wco, ::tensorflow::Input b,
              ::tensorflow::Input i, ::tensorflow::Input cs,
              ::tensorflow::Input f, ::tensorflow::Input o, ::tensorflow::Input
              ci, ::tensorflow::Input co, ::tensorflow::Input h,
              ::tensorflow::Input cs_grad, ::tensorflow::Input h_grad, bool
              use_peephole);

  Operation operation;
  ::tensorflow::Output x_grad;
  ::tensorflow::Output cs_prev_grad;
  ::tensorflow::Output h_prev_grad;
  ::tensorflow::Output w_grad;
  ::tensorflow::Output wci_grad;
  ::tensorflow::Output wcf_grad;
  ::tensorflow::Output wco_grad;
  ::tensorflow::Output b_grad;
};

/// Computes the LSTM cell forward propagation for 1 time step.
///
/// This implementation uses 1 weight matrix and 1 bias vector, and there's an
/// optional peephole connection.
///
/// This kernel op implements the following mathematical equations:
///
/// ```python
/// xh = [x, h_prev]
/// [i, f, ci, o] = xh * w + b
/// f = f + forget_bias
///
/// if not use_peephole:
///   wci = wcf = wco = 0
///
/// i = sigmoid(cs_prev * wci + i)
/// f = sigmoid(cs_prev * wcf + f)
/// ci = tanh(ci)
///
/// cs = ci .* i + cs_prev .* f
/// cs = clip(cs, cell_clip)
///
/// o = sigmoid(cs * wco + o)
/// co = tanh(cs)
/// h = co .* o
/// ```
///
/// Arguments:
/// * scope: A Scope object
/// * x: The input to the LSTM cell, shape (batch_size, num_inputs).
/// * cs_prev: Value of the cell state at previous time step.
/// * h_prev: Output of the previous cell at previous time step.
/// * w: The weight matrix.
/// * wci: The weight matrix for input gate peephole connection.
/// * wcf: The weight matrix for forget gate peephole connection.
/// * wco: The weight matrix for output gate peephole connection.
/// * b: The bias vector.
///
/// Optional attributes (see `Attrs`):
/// * forget_bias: The forget gate bias.
/// * cell_clip: Value to clip the 'cs' value to.
/// * use_peephole: Whether to use peephole weights.
///
/// Returns:
/// * `Output` i: The input gate.
/// * `Output` cs: The cell state before the tanh.
/// * `Output` f: The forget gate.
/// * `Output` o: The output gate.
/// * `Output` ci: The cell input.
/// * `Output` co: The cell after the tanh.
/// * `Output` h: The output h vector.
class LSTMBlockCell {
 public:
  /// Optional attribute setters for LSTMBlockCell
  struct Attrs {
    /// The forget gate bias.
    ///
    /// Defaults to 1
    TF_MUST_USE_RESULT Attrs ForgetBias(float x) {
      Attrs ret = *this;
      ret.forget_bias_ = x;
      return ret;
    }

    /// Value to clip the 'cs' value to.
    ///
    /// Defaults to 3
    TF_MUST_USE_RESULT Attrs CellClip(float x) {
      Attrs ret = *this;
      ret.cell_clip_ = x;
      return ret;
    }

    /// Whether to use peephole weights.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UsePeephole(bool x) {
      Attrs ret = *this;
      ret.use_peephole_ = x;
      return ret;
    }

    float forget_bias_ = 1.0f;
    float cell_clip_ = 3.0f;
    bool use_peephole_ = false;
  };
  LSTMBlockCell(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
              ::tensorflow::Input cs_prev, ::tensorflow::Input h_prev,
              ::tensorflow::Input w, ::tensorflow::Input wci,
              ::tensorflow::Input wcf, ::tensorflow::Input wco,
              ::tensorflow::Input b);
  LSTMBlockCell(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
              ::tensorflow::Input cs_prev, ::tensorflow::Input h_prev,
              ::tensorflow::Input w, ::tensorflow::Input wci,
              ::tensorflow::Input wcf, ::tensorflow::Input wco,
              ::tensorflow::Input b, const LSTMBlockCell::Attrs& attrs);

  static Attrs ForgetBias(float x) {
    return Attrs().ForgetBias(x);
  }
  static Attrs CellClip(float x) {
    return Attrs().CellClip(x);
  }
  static Attrs UsePeephole(bool x) {
    return Attrs().UsePeephole(x);
  }

  Operation operation;
  ::tensorflow::Output i;
  ::tensorflow::Output cs;
  ::tensorflow::Output f;
  ::tensorflow::Output o;
  ::tensorflow::Output ci;
  ::tensorflow::Output co;
  ::tensorflow::Output h;
};

/// Computes the LSTM cell backward propagation for 1 timestep.
///
/// This implementation is to be used in conjunction of LSTMBlockCell.
///
/// Arguments:
/// * scope: A Scope object
/// * x: The input to the LSTM cell, shape (batch_size, num_inputs).
/// * cs_prev: The previous cell state.
/// * h_prev: The previous h state.
/// * w: The weight matrix.
/// * wci: The weight matrix for input gate peephole connection.
/// * wcf: The weight matrix for forget gate peephole connection.
/// * wco: The weight matrix for output gate peephole connection.
/// * b: The bias vector.
/// * i: The input gate.
/// * cs: The cell state before the tanh.
/// * f: The forget gate.
/// * o: The output gate.
/// * ci: The cell input.
/// * co: The cell after the tanh.
/// * cs_grad: The current gradient of cs.
/// * h_grad: The gradient of h vector.
/// * use_peephole: Whether the cell uses peephole connections.
///
/// Returns:
/// * `Output` cs_prev_grad: The gradient of cs to be back-propped.
/// * `Output` dicfo: The derivative wrt to [i, cs, f, o].
/// * `Output` wci_grad: The gradient for wci to be back-propped.
/// * `Output` wcf_grad: The gradient for wcf to be back-propped.
/// * `Output` wco_grad: The gradient for wco to be back-propped.
class LSTMBlockCellGrad {
 public:
  LSTMBlockCellGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
                  ::tensorflow::Input cs_prev, ::tensorflow::Input h_prev,
                  ::tensorflow::Input w, ::tensorflow::Input wci,
                  ::tensorflow::Input wcf, ::tensorflow::Input wco,
                  ::tensorflow::Input b, ::tensorflow::Input i,
                  ::tensorflow::Input cs, ::tensorflow::Input f,
                  ::tensorflow::Input o, ::tensorflow::Input ci,
                  ::tensorflow::Input co, ::tensorflow::Input cs_grad,
                  ::tensorflow::Input h_grad, bool use_peephole);

  Operation operation;
  ::tensorflow::Output cs_prev_grad;
  ::tensorflow::Output dicfo;
  ::tensorflow::Output wci_grad;
  ::tensorflow::Output wcf_grad;
  ::tensorflow::Output wco_grad;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_LSTM_OPS_H_
