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

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "paddle/cinn/adt/anchor_sd_equation_context.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_graph.h"
#include "paddle/cinn/adt/m_ir.h"

namespace cinn::adt {

using AnchorIndex = eqaution::Index;
using EquationCtx4OpStmtT =
    std::function<std::shared_ptr<equation::config::OpEquationContext>(
        const m_expr::OpStmt&)>;

class IGroup final {
 public:
  IGroup(const IGroup&) = delete;
  IGroup(IGroup&&) = delete;

  explicit IGroup(const List<m_expr::OpStmt>& op_stmts,
                  const AnchorIndex& anchor_index,
                  const EquationCtx4OpStmtT& EquationCtx4OpStmt)
      : op_stmts_(op_stmts),
        anchor_index_(anchor_index),
        EquationCtx4OpStmt_(EquationCtx4OpStmt),
        index2tensor_(GenerateIndex2Tensor(op_stmts, EquationCtx4OpStmt)) {}

  const List<m_expr::OpStmt>& op_stmts() const { return op_stmts_; }

  const AnchorIndex& anchor_index() const { return anchor_index_; }

  const m_expr::Tensor& anchor_tensor() const {
    return GetTensor(anchor_index());
  }

  GraphView GetDefaultGraphView() const {
    return partition::MakeGlobalEquationGraphViewForPartition(
        EquationCtx4OpStmt_, op_stmts_);
  }

  const m_expr::Tensor& GetTensor(const Index& index) const {
    return index2tensor_->at(index);
  }

  const std::optional<equation::config::AnchorSdEquationContext>&
  anchor_sd_equation_ctx() const {
    return anchor_sd_equation_ctx_;
  }

  void set_anchor_sd_equation_ctx(
      const equation::config::AnchorSdEquationContext& ctx) {
    anchor_sd_equation_ctx_ = ctx;
  }

  const List<Iterator>& sd_iterators() const {
    CHECK(anchor_sd_equation_ctx_.has_value());
    return anchor_sd_equation_ctx_.value().sd_iterators();
  }

 private:
  static std::unordered_map<eqaution::Index, m_expr::Tensor>
  GenerateIndex2Tensor(const List<m_expr::OpStmt>& op_stmts,
                       const EquationCtx4OpStmtT& EquationCtx4OpStmt) {
    std::unordered_map<eqaution::Index, m_expr::Tensor> index2tensor;

    for (const auto& op_stmt : *op_stmts) {
      const auto* ctx = EquationCtx4OpStmt(op_stmt);
      const auto& [op, op_inputs, op_outputs] = op_stmt.tuple();
      for (std::size_t idx = 0; idx < op_inputs.value()->size(); ++idx) {
        CHECK(index2tensor
                  .emplace(ctx->GetInIndex(idx), op_inputs.value()->at(idx))
                  .second);
      }
      for (std::size_t idx = 0; idx < op_outputs.value()->size(); ++idx) {
        CHECK(index2tensor
                  .emplace(ctx->GetOutIndex(idx), op_outputs.value()->at(idx))
                  .second);
      }
    }

    return index2tensor;
  }

  List<m_expr::OpStmt> op_stmts_;
  AnchorIndex anchor_index_;
  EquationCtx4OpStmtT EquationCtx4OpStmt_;
  std::unordered_map<equation::Index, m_expr::Tensor> index2tensor_;
  std::optional<equation::config::AnchorSdEquationContext>
      anchor_sd_equation_ctx_;
};

}  // namespace cinn::adt
