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

#include "paddle/cinn/adt/print_constant.h"
#include "paddle/cinn/adt/print_equations.h"
#include "paddle/cinn/adt/print_map_expr.h"
#include "paddle/cinn/adt/print_schedule_descriptor.h"
#include "paddle/cinn/adt/print_value.h"

namespace cinn::adt {

// print_constant.h
std::string ToTxtString(const Constant& constant);

// print_map_expr.h
std::string ToTxtString(const MapExpr& map_expr, const std::string& group_id);

// print_equations.h
std::string ToTxtString(const Equation& equation);

std::string ToTxtString(const Equations& equations,
                        const std::string& separator = "\n");

std::string ToTxtString(const Iterator& iterator);

std::string ToTxtString(const Index& index);

std::string ToTxtString(const FakeOpPlaceHolder& op);

std::string ToTxtString(const List<Index>& indexes);

std::string ToTxtString(const List<std::optional<Index>>& indexes);

std::string ToTxtString(const List<Stride>& strides);

std::string ToTxtString(const List<Iterator>& iterators);

std::string ToTxtString(const tInMsgBox<List<Index>>& in_msg_box_indexes);

std::string ToTxtString(const tOutMsgBox<List<Index>>& out_msg_box_indexes);

std::string ToTxtString(const std::vector<Index>& indexes);

std::string ToTxtString(const List<OpStmt>& op_stmts,
                        const EquationCtx4OpStmtT& EquationCtx4OpStmt);

// print_schedule_descriptor.h
std::string ToTxtString(const LoopDescriptor& loop_descriptor);

// print_value.h
std::string ToTxtString(const Value& value);

std::string ToTxtString(const std::optional<Value>& opt_value);

}  // namespace cinn::adt
