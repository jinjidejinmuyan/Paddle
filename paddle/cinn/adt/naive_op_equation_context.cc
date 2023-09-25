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

namespace cinn::adt::config {

namespace {

using InBox2OutBox = InMsgBox2OutMsgBox<tOut<FakeOpPlaceHolder>,
                                        tOut<tOutMsgBox<OpArgIndexes>>,
                                        tIn<tInMsgBox<OpArgIndexes>>>;

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

List<Index> GetNonErasedIndexes(const List<Index>& origin_indexes,
                                const std::vector<Index>& erased_indexes) {
  List<Index> indexes{};
  for (const auto& index : *origin_indexes) {
    if (std::find(erased_indexes.begin(), erased_indexes.end(), index) ==
        erased_indexes.end()) {
      indexes->emplace_back(index);
    }
  }
  return indexes;
}

Equation EraseIndexes(const Equation& equation,
                      const std::vector<Index>& erased_output_tensor_indexes) {
  const auto& in_msg_box2out_msg_box = equation.Get<InBox2OutBox>();
  const auto& [op_placeholder, out_box_indexes, in_box_indexes] =
      in_msg_box2out_msg_box.tuple();
  const auto& [out_box_in_indexes, out_box_out_indexes] =
      out_box_indexes.value().value().tuple();
  const auto& non_erased_out_box_out_indexes = GetNonErasedIndexes(
      out_box_out_indexes.value(), erased_output_tensor_indexes);
  return InBox2OutBox{op_placeholder,
                      tOut<tOutMsgBox<OpArgIndexes>>{OpArgIndexes{
                          out_box_in_indexes, non_erased_out_box_out_indexes}},
                      in_box_indexes};
}

}  // namespace

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
