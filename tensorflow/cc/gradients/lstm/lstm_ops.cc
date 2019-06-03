// This file is MACHINE GENERATED! Do not edit.


#include "tensorflow/cc/ops/const_op.h"
#include "lstm_ops.h"

namespace tensorflow {
namespace ops {

BlockLSTM::BlockLSTM(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     seq_len_max, ::tensorflow::Input x, ::tensorflow::Input
                     cs_prev, ::tensorflow::Input h_prev, ::tensorflow::Input
                     w, ::tensorflow::Input wci, ::tensorflow::Input wcf,
                     ::tensorflow::Input wco, ::tensorflow::Input b, const
                     BlockLSTM::Attrs& attrs) {
  if (!scope.ok()) return;
  auto _seq_len_max = ::tensorflow::ops::AsNodeOut(scope, seq_len_max);
  if (!scope.ok()) return;
  auto _x = ::tensorflow::ops::AsNodeOut(scope, x);
  if (!scope.ok()) return;
  auto _cs_prev = ::tensorflow::ops::AsNodeOut(scope, cs_prev);
  if (!scope.ok()) return;
  auto _h_prev = ::tensorflow::ops::AsNodeOut(scope, h_prev);
  if (!scope.ok()) return;
  auto _w = ::tensorflow::ops::AsNodeOut(scope, w);
  if (!scope.ok()) return;
  auto _wci = ::tensorflow::ops::AsNodeOut(scope, wci);
  if (!scope.ok()) return;
  auto _wcf = ::tensorflow::ops::AsNodeOut(scope, wcf);
  if (!scope.ok()) return;
  auto _wco = ::tensorflow::ops::AsNodeOut(scope, wco);
  if (!scope.ok()) return;
  auto _b = ::tensorflow::ops::AsNodeOut(scope, b);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("BlockLSTM");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "BlockLSTM")
                     .Input(_seq_len_max)
                     .Input(_x)
                     .Input(_cs_prev)
                     .Input(_h_prev)
                     .Input(_w)
                     .Input(_wci)
                     .Input(_wcf)
                     .Input(_wco)
                     .Input(_b)
                     .Attr("forget_bias", attrs.forget_bias_)
                     .Attr("cell_clip", attrs.cell_clip_)
                     .Attr("use_peephole", attrs.use_peephole_)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->operation = Operation(ret);
  ::tensorflow::NameRangeMap _outputs_range;
  ::tensorflow::Status _status_ = ::tensorflow::NameRangesForNode(*ret, ret->op_def(), nullptr, &_outputs_range);
  if (!_status_.ok()) {
    scope.UpdateStatus(_status_);
    return;
  }

  this->i = Output(ret, _outputs_range["i"].first);
  this->cs = Output(ret, _outputs_range["cs"].first);
  this->f = Output(ret, _outputs_range["f"].first);
  this->o = Output(ret, _outputs_range["o"].first);
  this->ci = Output(ret, _outputs_range["ci"].first);
  this->co = Output(ret, _outputs_range["co"].first);
  this->h = Output(ret, _outputs_range["h"].first);
}

BlockLSTM::BlockLSTM(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     seq_len_max, ::tensorflow::Input x, ::tensorflow::Input
                     cs_prev, ::tensorflow::Input h_prev, ::tensorflow::Input
                     w, ::tensorflow::Input wci, ::tensorflow::Input wcf,
                     ::tensorflow::Input wco, ::tensorflow::Input b)
  : BlockLSTM(scope, seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b, BlockLSTM::Attrs()) {}

BlockLSTMGrad::BlockLSTMGrad(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input seq_len_max,
                             ::tensorflow::Input x, ::tensorflow::Input
                             cs_prev, ::tensorflow::Input h_prev,
                             ::tensorflow::Input w, ::tensorflow::Input wci,
                             ::tensorflow::Input wcf, ::tensorflow::Input wco,
                             ::tensorflow::Input b, ::tensorflow::Input i,
                             ::tensorflow::Input cs, ::tensorflow::Input f,
                             ::tensorflow::Input o, ::tensorflow::Input ci,
                             ::tensorflow::Input co, ::tensorflow::Input h,
                             ::tensorflow::Input cs_grad, ::tensorflow::Input
                             h_grad, bool use_peephole) {
  if (!scope.ok()) return;
  auto _seq_len_max = ::tensorflow::ops::AsNodeOut(scope, seq_len_max);
  if (!scope.ok()) return;
  auto _x = ::tensorflow::ops::AsNodeOut(scope, x);
  if (!scope.ok()) return;
  auto _cs_prev = ::tensorflow::ops::AsNodeOut(scope, cs_prev);
  if (!scope.ok()) return;
  auto _h_prev = ::tensorflow::ops::AsNodeOut(scope, h_prev);
  if (!scope.ok()) return;
  auto _w = ::tensorflow::ops::AsNodeOut(scope, w);
  if (!scope.ok()) return;
  auto _wci = ::tensorflow::ops::AsNodeOut(scope, wci);
  if (!scope.ok()) return;
  auto _wcf = ::tensorflow::ops::AsNodeOut(scope, wcf);
  if (!scope.ok()) return;
  auto _wco = ::tensorflow::ops::AsNodeOut(scope, wco);
  if (!scope.ok()) return;
  auto _b = ::tensorflow::ops::AsNodeOut(scope, b);
  if (!scope.ok()) return;
  auto _i = ::tensorflow::ops::AsNodeOut(scope, i);
  if (!scope.ok()) return;
  auto _cs = ::tensorflow::ops::AsNodeOut(scope, cs);
  if (!scope.ok()) return;
  auto _f = ::tensorflow::ops::AsNodeOut(scope, f);
  if (!scope.ok()) return;
  auto _o = ::tensorflow::ops::AsNodeOut(scope, o);
  if (!scope.ok()) return;
  auto _ci = ::tensorflow::ops::AsNodeOut(scope, ci);
  if (!scope.ok()) return;
  auto _co = ::tensorflow::ops::AsNodeOut(scope, co);
  if (!scope.ok()) return;
  auto _h = ::tensorflow::ops::AsNodeOut(scope, h);
  if (!scope.ok()) return;
  auto _cs_grad = ::tensorflow::ops::AsNodeOut(scope, cs_grad);
  if (!scope.ok()) return;
  auto _h_grad = ::tensorflow::ops::AsNodeOut(scope, h_grad);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("BlockLSTMGrad");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "BlockLSTMGrad")
                     .Input(_seq_len_max)
                     .Input(_x)
                     .Input(_cs_prev)
                     .Input(_h_prev)
                     .Input(_w)
                     .Input(_wci)
                     .Input(_wcf)
                     .Input(_wco)
                     .Input(_b)
                     .Input(_i)
                     .Input(_cs)
                     .Input(_f)
                     .Input(_o)
                     .Input(_ci)
                     .Input(_co)
                     .Input(_h)
                     .Input(_cs_grad)
                     .Input(_h_grad)
                     .Attr("use_peephole", use_peephole)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->operation = Operation(ret);
  ::tensorflow::NameRangeMap _outputs_range;
  ::tensorflow::Status _status_ = ::tensorflow::NameRangesForNode(*ret, ret->op_def(), nullptr, &_outputs_range);
  if (!_status_.ok()) {
    scope.UpdateStatus(_status_);
    return;
  }

  this->x_grad = Output(ret, _outputs_range["x_grad"].first);
  this->cs_prev_grad = Output(ret, _outputs_range["cs_prev_grad"].first);
  this->h_prev_grad = Output(ret, _outputs_range["h_prev_grad"].first);
  this->w_grad = Output(ret, _outputs_range["w_grad"].first);
  this->wci_grad = Output(ret, _outputs_range["wci_grad"].first);
  this->wcf_grad = Output(ret, _outputs_range["wcf_grad"].first);
  this->wco_grad = Output(ret, _outputs_range["wco_grad"].first);
  this->b_grad = Output(ret, _outputs_range["b_grad"].first);
}

LSTMBlockCell::LSTMBlockCell(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input x, ::tensorflow::Input
                             cs_prev, ::tensorflow::Input h_prev,
                             ::tensorflow::Input w, ::tensorflow::Input wci,
                             ::tensorflow::Input wcf, ::tensorflow::Input wco,
                             ::tensorflow::Input b, const LSTMBlockCell::Attrs&
                             attrs) {
  if (!scope.ok()) return;
  auto _x = ::tensorflow::ops::AsNodeOut(scope, x);
  if (!scope.ok()) return;
  auto _cs_prev = ::tensorflow::ops::AsNodeOut(scope, cs_prev);
  if (!scope.ok()) return;
  auto _h_prev = ::tensorflow::ops::AsNodeOut(scope, h_prev);
  if (!scope.ok()) return;
  auto _w = ::tensorflow::ops::AsNodeOut(scope, w);
  if (!scope.ok()) return;
  auto _wci = ::tensorflow::ops::AsNodeOut(scope, wci);
  if (!scope.ok()) return;
  auto _wcf = ::tensorflow::ops::AsNodeOut(scope, wcf);
  if (!scope.ok()) return;
  auto _wco = ::tensorflow::ops::AsNodeOut(scope, wco);
  if (!scope.ok()) return;
  auto _b = ::tensorflow::ops::AsNodeOut(scope, b);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("LSTMBlockCell");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "LSTMBlockCell")
                     .Input(_x)
                     .Input(_cs_prev)
                     .Input(_h_prev)
                     .Input(_w)
                     .Input(_wci)
                     .Input(_wcf)
                     .Input(_wco)
                     .Input(_b)
                     .Attr("forget_bias", attrs.forget_bias_)
                     .Attr("cell_clip", attrs.cell_clip_)
                     .Attr("use_peephole", attrs.use_peephole_)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->operation = Operation(ret);
  ::tensorflow::NameRangeMap _outputs_range;
  ::tensorflow::Status _status_ = ::tensorflow::NameRangesForNode(*ret, ret->op_def(), nullptr, &_outputs_range);
  if (!_status_.ok()) {
    scope.UpdateStatus(_status_);
    return;
  }

  this->i = Output(ret, _outputs_range["i"].first);
  this->cs = Output(ret, _outputs_range["cs"].first);
  this->f = Output(ret, _outputs_range["f"].first);
  this->o = Output(ret, _outputs_range["o"].first);
  this->ci = Output(ret, _outputs_range["ci"].first);
  this->co = Output(ret, _outputs_range["co"].first);
  this->h = Output(ret, _outputs_range["h"].first);
}

LSTMBlockCell::LSTMBlockCell(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input x, ::tensorflow::Input
                             cs_prev, ::tensorflow::Input h_prev,
                             ::tensorflow::Input w, ::tensorflow::Input wci,
                             ::tensorflow::Input wcf, ::tensorflow::Input wco,
                             ::tensorflow::Input b)
  : LSTMBlockCell(scope, x, cs_prev, h_prev, w, wci, wcf, wco, b, LSTMBlockCell::Attrs()) {}

LSTMBlockCellGrad::LSTMBlockCellGrad(const ::tensorflow::Scope& scope,
                                     ::tensorflow::Input x, ::tensorflow::Input
                                     cs_prev, ::tensorflow::Input h_prev,
                                     ::tensorflow::Input w, ::tensorflow::Input
                                     wci, ::tensorflow::Input wcf,
                                     ::tensorflow::Input wco,
                                     ::tensorflow::Input b, ::tensorflow::Input
                                     i, ::tensorflow::Input cs,
                                     ::tensorflow::Input f, ::tensorflow::Input
                                     o, ::tensorflow::Input ci,
                                     ::tensorflow::Input co,
                                     ::tensorflow::Input cs_grad,
                                     ::tensorflow::Input h_grad, bool
                                     use_peephole) {
  if (!scope.ok()) return;
  auto _x = ::tensorflow::ops::AsNodeOut(scope, x);
  if (!scope.ok()) return;
  auto _cs_prev = ::tensorflow::ops::AsNodeOut(scope, cs_prev);
  if (!scope.ok()) return;
  auto _h_prev = ::tensorflow::ops::AsNodeOut(scope, h_prev);
  if (!scope.ok()) return;
  auto _w = ::tensorflow::ops::AsNodeOut(scope, w);
  if (!scope.ok()) return;
  auto _wci = ::tensorflow::ops::AsNodeOut(scope, wci);
  if (!scope.ok()) return;
  auto _wcf = ::tensorflow::ops::AsNodeOut(scope, wcf);
  if (!scope.ok()) return;
  auto _wco = ::tensorflow::ops::AsNodeOut(scope, wco);
  if (!scope.ok()) return;
  auto _b = ::tensorflow::ops::AsNodeOut(scope, b);
  if (!scope.ok()) return;
  auto _i = ::tensorflow::ops::AsNodeOut(scope, i);
  if (!scope.ok()) return;
  auto _cs = ::tensorflow::ops::AsNodeOut(scope, cs);
  if (!scope.ok()) return;
  auto _f = ::tensorflow::ops::AsNodeOut(scope, f);
  if (!scope.ok()) return;
  auto _o = ::tensorflow::ops::AsNodeOut(scope, o);
  if (!scope.ok()) return;
  auto _ci = ::tensorflow::ops::AsNodeOut(scope, ci);
  if (!scope.ok()) return;
  auto _co = ::tensorflow::ops::AsNodeOut(scope, co);
  if (!scope.ok()) return;
  auto _cs_grad = ::tensorflow::ops::AsNodeOut(scope, cs_grad);
  if (!scope.ok()) return;
  auto _h_grad = ::tensorflow::ops::AsNodeOut(scope, h_grad);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("LSTMBlockCellGrad");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "LSTMBlockCellGrad")
                     .Input(_x)
                     .Input(_cs_prev)
                     .Input(_h_prev)
                     .Input(_w)
                     .Input(_wci)
                     .Input(_wcf)
                     .Input(_wco)
                     .Input(_b)
                     .Input(_i)
                     .Input(_cs)
                     .Input(_f)
                     .Input(_o)
                     .Input(_ci)
                     .Input(_co)
                     .Input(_cs_grad)
                     .Input(_h_grad)
                     .Attr("use_peephole", use_peephole)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->operation = Operation(ret);
  ::tensorflow::NameRangeMap _outputs_range;
  ::tensorflow::Status _status_ = ::tensorflow::NameRangesForNode(*ret, ret->op_def(), nullptr, &_outputs_range);
  if (!_status_.ok()) {
    scope.UpdateStatus(_status_);
    return;
  }

  this->cs_prev_grad = Output(ret, _outputs_range["cs_prev_grad"].first);
  this->dicfo = Output(ret, _outputs_range["dicfo"].first);
  this->wci_grad = Output(ret, _outputs_range["wci_grad"].first);
  this->wcf_grad = Output(ret, _outputs_range["wcf_grad"].first);
  this->wco_grad = Output(ret, _outputs_range["wco_grad"].first);
}

/// @}

}  // namespace ops
}  // namespace tensorflow
