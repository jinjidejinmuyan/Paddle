// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>

#include "paddle/cinn/adt/adapter.h"
#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"
#include "paddle/cinn/adt/print_equations.h"

namespace cinn::adt::config {

namespace {

using InBox2OutBox =
    InMsgBox2OutMsgBox<tOut<FakeOpPlaceHolder>,
                       tOut<OpArgIndexes<std::optional<Index>>>,
                       tIn<OpArgIndexes<Index>>>;

template <typename HandleInMsgBox2OutMsgBoxT, typename HandleOtherT>
Equations TransformEquations(
    const Equations& origin_equations,
    const HandleInMsgBox2OutMsgBoxT& HandleInMsgBox2OutMsgBox,
    const HandleOtherT& HandleOther) {
  Equations equations{};
  for (const auto& origin_equation : *origin_equations) {
    if (origin_equation.Has<InBox2OutBox>()) {
      equations->emplace_back(HandleInMsgBox2OutMsgBox(origin_equation));
    } else {
      equations->emplace_back(HandleOther(origin_equation));
    }
  }
  return equations;
}

List<std::optional<Index>> GetMaskedOutIndexes(
    const List<Index>& in_box_out_indexes,
    const List<std::optional<Index>>& out_box_out_indexes,
    const std::vector<Index>& erased_in_msg_box_out_tensor_indexes) {
  List<std::optional<Index>> ret{};
  const auto& erased = erased_in_msg_box_out_tensor_indexes;
  CHECK_EQ(in_box_out_indexes->size(), out_box_out_indexes->size());
  for (std::size_t i = 0; i < in_box_out_indexes->size(); ++i) {
    const auto& in_box_index = in_box_out_indexes->at(i);
    if (std::find(erased.begin(), erased.end(), in_box_index) == erased.end()) {
      ret->emplace_back(out_box_out_indexes->at(i));
    } else {
      ret->emplace_back(std::nullopt);
    }
  }
  return ret;
}

Equation EraseIndexes(
    const Equation& equation,
    const std::vector<Index>& erased_in_msg_box_out_tensor_indexes) {
  VLOG(3) << "origin-equation: " << ToTxtString(equation);
  VLOG(3) << "erased_output_tensor_indexes: ";
  PrintIndexVector(erased_in_msg_box_out_tensor_indexes);
  const auto& in_msg_box2out_msg_box = equation.Get<InBox2OutBox>();
  const auto& [op_placeholder, out_box_indexes, in_box_indexes] =
      in_msg_box2out_msg_box.tuple();

  const auto& [_, in_box_out_indexes] = in_box_indexes.value().tuple();
  const auto& [out_box_in_indexes, out_box_out_indexes] =
      out_box_indexes.value().tuple();
  const auto& masked_out_indexes =
      GetMaskedOutIndexes(in_box_out_indexes.value(),
                          out_box_out_indexes.value(),
                          erased_in_msg_box_out_tensor_indexes);

  OpArgIndexes<std::optional<Index>> out_box{out_box_in_indexes,
                                             masked_out_indexes};

  Equation ret_equation = InBox2OutBox{op_placeholder, out_box, in_box_indexes};
  VLOG(3) << "ret-equation: " << ToTxtString(ret_equation);
  return ret_equation;
}

}  // namespace

void NaiveOpEquationContext::Print() {
  VLOG(3) << "equations : \n" << ToTxtString(equations(), "\n");
}

void NaiveOpEquationContext::EraseOutMsgBoxIndexes(
    const std::vector<Index>& erased_output_tensor_indexes) {
  const auto& Identity = [](const Equation& equation) { return equation; };
  const auto& Erase = [&](const Equation& equation) {
    return EraseIndexes(equation, erased_output_tensor_indexes);
  };
  equations_ = TransformEquations(equations_, Erase, Identity);
}

std::vector<std::uint64_t> MakeTensorRanks(const List<Arg>& arg_lists) {
  std::vector<std::uint64_t> ret;
  for (const auto& arg : *arg_lists) {
    CHECK(arg.Has<adapter::Tensor>());
    ret.push_back(arg.Get<adapter::Tensor>().GetRank());
  }
  return ret;
}

void GenerateOpEquationsImpl(const hlir::framework::Node* op_node,
                             const OpStmt& op_stmt,
                             config::NaiveOpEquationContext* ctx) {
  const auto& [_, inputs, outputs] = op_stmt.tuple();

  using GenerateEquationFunc =
      std::function<void(config::OpEquationContext * ctx)>;

  const auto& generate_equations =
      hlir::framework::Operator::GetAttrs<GenerateEquationFunc>(
          "generate_equations");
  CHECK(generate_equations.Find(op_node->op()));
  generate_equations[op_node->op()](ctx);
}

void GenerateOpEquationsImpl(
    const tReduceAcc<const hlir::framework::Node*>& op_node,
    const OpStmt& op_stmt,
    config::NaiveOpEquationContext* ctx) {
  GenerateOpEquationsImpl(op_node.value(), op_stmt, ctx);
}

void GenerateOpEquationsImpl(
    const tReduceInit<const hlir::framework::Node*>& op_node,
    const OpStmt& op_stmt,
    config::NaiveOpEquationContext* ctx) {
  // Do nothing
}

void GenerateOpEquations(const OpStmt& op_stmt,
                         config::NaiveOpEquationContext* ctx) {
  const auto& [op, inputs, outputs] = op_stmt.tuple();

  return std::visit(
      [&](const auto& impl) {
        return GenerateOpEquationsImpl(impl, op_stmt, ctx);
      },
      op.variant());
}

const hlir::framework::AttrMapType* GetOpAttrImpl(
    const hlir::framework::Node* op_node) {
  return &op_node->attrs.attr_store;
}

const hlir::framework::AttrMapType* GetOpAttrImpl(
    const tReduceInit<const hlir::framework::Node*>&) {
  static hlir::framework::AttrMapType empty{};
  return &empty;
}

const hlir::framework::AttrMapType* GetOpAttrImpl(
    const tReduceAcc<const hlir::framework::Node*>& op_node) {
  return GetOpAttrImpl(op_node.value());
}

const hlir::framework::AttrMapType* GetOpAttr(const OpStmt& op_stmt) {
  const auto& [op_node, inputs, outputs] = op_stmt.tuple();

  const auto* attr = std::visit(
      [&](const auto& impl) { return GetOpAttrImpl(impl); }, op_node.variant());

  return attr;
}

std::shared_ptr<config::NaiveOpEquationContext> MakeContextAndGenerateEquations(
    const OpStmt& op_stmt) {
  const auto& [op, inputs, outputs] = op_stmt.tuple();
  const auto& ctx = std::make_shared<config::NaiveOpEquationContext>(
      MakeTensorRanks(inputs.value()),
      MakeTensorRanks(outputs.value()),
      GetOpAttr(op_stmt));

  GenerateOpEquations(op_stmt, ctx.get());

  return ctx;
}

std::function<std::shared_ptr<config::NaiveOpEquationContext>(const OpStmt&)>
GenerateContext4LocalOpStmt(const List<OpStmt>& op_stmts) {
  using OpStmt2EquationContext =
      std::unordered_map<OpStmt,
                         std::shared_ptr<config::NaiveOpEquationContext>>;
  const auto& op_stmt2equation_ctx = std::make_shared<OpStmt2EquationContext>();

  for (const auto& op_stmt : *op_stmts) {
    const auto& ctx = MakeContextAndGenerateEquations(op_stmt);
    CHECK(op_stmt2equation_ctx->emplace(op_stmt, ctx).second);
  }

  return [op_stmt2equation_ctx](const auto& op_stmt) {
    return op_stmt2equation_ctx->at(op_stmt);
  };
}

}  // namespace cinn::adt::config
