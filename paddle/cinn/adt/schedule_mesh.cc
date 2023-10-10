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

#include "paddle/cinn/adt/schedule_mesh.h"

namespace cinn::adt {

namespace {

std::size_t GetInputRankImpl(const List<ScheduleDim>& sched_dims) {
  return sched_dims->size();
}

std::size_t GetInputRankImpl(
    const ScheduleMeshReshape<ScheduleMesh>& sched_reshape) {
  const auto& [sched_mesh, _] = sched_reshape.tuple();
  return GetInputRank(sched_mesh);
}

std::size_t GetInputRankImpl(
    const ScheduleMeshTranspose<ScheduleMesh>& sched_transpose) {
  const auto& [sched_mesh, _] = sched_transpose.tuple();
  return GetInputRank(sched_mesh);
}

std::size_t GetInputRankImpl(
    const ScheduleMeshPadding<ScheduleMesh>& sched_padding) {
  const auto& [sched_mesh, _] = sched_padding.tuple();
  return GetInputRank(sched_mesh);
}

}  // namespace

std::size_t GetInputRank(const ScheduleMesh& sched_mesh) {
  return std::visit([&](const auto& impl) { return GetInputRankImpl(impl); },
                    sched_mesh.variant());
}

namespace {

std::size_t GetOutputRankImpl(const List<ScheduleDim>& sched_dims) {
  return sched_dims->size();
}

std::size_t GetOutputRankImpl(
    const ScheduleMeshReshape<ScheduleMesh>& sched_reshape) {
  const auto& [_, shapes] = sched_reshape.tuple();
  return shapes.value()->size();
}

std::size_t GetOutputRankImpl(
    const ScheduleMeshTranspose<ScheduleMesh>& sched_transpose) {
  const auto& [sched_mesh, perm] = sched_transpose.tuple();
  CHECK_EQ(GetRank(sched_mesh), perm.value()->size());
  return perm.value()->size();
}

std::size_t GetOutputRankImpl(
    const ScheduleMeshPadding<ScheduleMesh>& sched_padding) {
  const auto& [_, padding_to] = sched_padding.tuple();
  return padding_to.value()->size();
}

}  // namespace

std::size_t GetOutputRank(const ScheduleMesh& sched_mesh) {
  return std::visit([&](const auto& impl) { return GetOutputRankImpl(impl); },
                    sched_mesh.variant());
}

namespace {

List<Constant> GetOutputDimValuesImpl(const List<ScheduleDim>& sched_dims) {
  List<Constant> ret{};
  for (const auto& sched_dim : *sched_dims) {
    const auto& loop_size = GetLoopSize(sched_dim);
    CHECK(loop_size.Has<std::int64_t>());
    ret->emplace_back(loop_size.Get<std::int64_t>());
  }
  return ret;
}

List<Constant> GetOutputDimValuesImpl(
    const ScheduleMeshReshape<ScheduleMesh>& sched_reshape) {
  const auto& [_, shape] = sched_reshape.tuple();
  List<Constant> ret{};
  for (const auto& dim : *shape.value()) {
    const auto& loop_size = GetLoopSize(dim);
    CHECK(loop_size.Has<std::int64_t>());
    ret->emplace_back(loop_size.Get<std::int64_t>());
  }
  return ret;
}

List<Constant> GetOutputDimValuesImpl(
    const ScheduleMeshTranspose<ScheduleMesh>& sched_transpose) {
  const auto& [sched_mesh, perm] = sched_transpose.tuple();
  const auto& input_dims = GetOutputDimValues(sched_mesh);
  List<Constant> ret{};
  for (const auto& idx : *perm.value()) {
    ret->emplace_back(input_dims->at(idx));
  }
  return ret;
}

List<Constant> GetOutputDimValuesImpl(
    const ScheduleMeshPadding<ScheduleMesh>& sched_padding) {
  const auto& [_, shape] = sched_padding.tuple();
  List<Constant> ret{};
  for (const auto& dim : *shape.value()) {
    const auto& loop_size = GetLoopSize(dim);
    CHECK(loop_size.Has<std::int64_t>());
    ret->emplace_back(loop_size.Get<std::int64_t>());
  }
  return ret;
}

}  // namespace

List<Constant> GetOutputDimValues(const ScheduleMesh& sched_mesh) {
  return std::visit(
      [&](const auto& impl) { return GetOutputDimValuesImpl(impl); },
      sched_mesh.variant());
}

List<Constant> GetOutputStrideValues(const ScheduleMesh& sched_mesh) {
  const auto& dims = GetOutputDimValues(sched_mesh);
  std::int64_t acc = 1;
  List<Constant> ret{};
  for (int i = dims->size() - 1; i >= 0; --i) {
    ret->emplace_back(acc);
    CHECK(dims->at(i).Has<std::int64_t>());
    acc *= dims->at(i).Get<std::int64_t>();
  }
  std::reverse(ret->begin(), ret->end());
  return ret;
}

namespace {

ScheduleMesh GetInputScheduleMeshImpl(const List<ScheduleDim>& sched_dims) {
  return sched_dims;
}

ScheduleMesh GetInputScheduleMeshImpl(
    const ScheduleMeshReshape<ScheduleMesh>& sched_reshape) {
  const auto& [sched_mesh, _] = sched_reshape.tuple();
  return GetInputScheduleMesh(sched_mesh);
}

ScheduleMesh GetInputScheduleMeshImpl(
    const ScheduleMeshTranspose<ScheduleMesh>& sched_transpose) {
  const auto& [sched_mesh, _] = sched_transpose.tuple();
  return GetInputScheduleMesh(sched_mesh);
}

ScheduleMesh GetInputScheduleMeshImpl(
    const ScheduleMeshPadding<ScheduleMesh>& sched_padding) {
  const auto& [sched_mesh, _] = sched_padding.tuple();
  return GetInputScheduleMesh(sched_mesh);
}

}  // namespace

ScheduleMesh GetInputScheduleMesh(const ScheduleMesh& sched_mesh) {
  return std::visit(
      [&](const auto& impl) { return GetInputScheduleMeshImpl(impl); },
      sched_mesh.variant());
}

namespace {

ScheduleMesh GenerateTransposeScheduleMesh(const ScheduleMesh& sched_mesh) {
  const auto& sched_dims = GetOutputDimValues(sched_mesh);
  const auto& is_reduce
}

}  // namespace

std::tuple<ScheduleMesh, List<LoopType>> CreateOptimizedScheduleMesh(
    const List<ScheduleDim>& loop_sizes) {
  ADT_TODO();
}

}  // namespace cinn::adt
