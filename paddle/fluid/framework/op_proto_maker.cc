/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/operators/ops_extra_info.h"

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

void OpProtoAndCheckerMaker::Validate() {
  validated_ = true;
  CheckNoDuplicatedInOutAttrs();
}

OpProtoAndCheckerMaker::VariableBuilder OpProtoAndCheckerMaker::AddInput(
    const std::string& name, const std::string& comment) {
  auto* input = proto_->add_inputs();
  input->set_name(name);
  input->set_comment(comment);
  return OpProtoAndCheckerMaker::VariableBuilder{input};
}

OpProtoAndCheckerMaker::VariableBuilder OpProtoAndCheckerMaker::AddOutput(
    const std::string& name, const std::string& comment) {
  auto* output = proto_->add_outputs();
  output->set_name(name);
  output->set_comment(comment);
  return OpProtoAndCheckerMaker::VariableBuilder{output};
}

void OpProtoAndCheckerMaker::CheckNoDuplicatedInOutAttrs() {
  std::unordered_set<std::string> names;
  auto checker = [&](const std::string& name) {
    PADDLE_ENFORCE_EQ(
        names.count(name),
        0,
        platform::errors::AlreadyExists("Attribute [%s] is duplicated.", name));
    names.insert(name);
  };
  for (auto& attr : proto_->attrs()) {
    checker(attr.name());
  }
  for (auto& input : proto_->inputs()) {
    checker(input.name());
  }
  for (auto& output : proto_->outputs()) {
    checker(output.name());
  }
}

// 调用示例：
// CustomOpMaker custom_maker(op_inputs, op_outputs, op_attrs);
// custom_maker(info.proto_, info.checker_);
// 然后会调用到这个函数，生成 op
void OpProtoAndCheckerMaker::operator()(proto::OpProto* proto,
                                        OpAttrChecker* attr_checker) {
  proto_ = proto;
  op_checker_ = attr_checker;
  // 此处调用每个 OP 定义时的 Make 函数，将 Input、Attr、Output 等放到 proto
  // 里面
  Make();
  op_checker_->RecordExplicitCheckerNum();

  const AttributeMap* extra_attrs_ptr = nullptr;
  const std::string& op_type = proto->type();
  // 一些 op 的 Extra 属性，放到这里来初始化（OP 规范化内容）
  const auto& extra_attr_map =
      operators::ExtraInfoUtils::Instance().GetExtraAttrsMap(op_type);
  if (!extra_attr_map.empty()) {
    extra_attrs_ptr = &extra_attr_map;
  }
  op_checker_->InitDefaultAttributeMap(extra_attrs_ptr);

  // 每个 Op
  // 都有一些必填的属性，【疑问】这些属性作用是什么？何时会修改这些属性？
  // implicit attribute, we mean the attribute added outside of the Make
  // method like "op_role", "op_role_var", and they are useless in dynamic
  // graph mode
  // See：paddle/fluid/framework/attribute_checker.h，the comment of
  // `explicit_checker_num_`
  AddAttr<int>(OpRoleAttrName(), "The role of this operator")
      .InEnum(
          {static_cast<int>(OpRole::kForward),
           static_cast<int>(OpRole::kBackward),
           static_cast<int>(OpRole::kOptimize),
           static_cast<int>(OpRole::kRPC),
           static_cast<int>(OpRole::kDist),
           static_cast<int>(OpRole::kLRSched),
           static_cast<int>(OpRole::kLoss) | static_cast<int>(OpRole::kForward),
           static_cast<int>(OpRole::kLoss) |
               static_cast<int>(OpRole::kBackward),
           static_cast<int>(OpRole::kOptimize) |
               static_cast<int>(OpRole::kLRSched),
           static_cast<int>(OpRole::kNotSpecified)})
      .SetDefault(static_cast<int>(OpRole::kNotSpecified))
      .AsExtra();
  AddAttr<std::vector<std::string>>(OpRoleVarAttrName(),
                                    "Optimized for variable")
      .SetDefault({})
      .AsExtra();

  AddAttr<std::string>(OpNamescopeAttrName(), "Operator name with namescope.")
      .SetDefault("")
      .AsExtra();

  AddAttr<std::vector<std::string>>(OpCreationCallstackAttrName(),
                                    "Callstack for Op Creation.")
      .SetDefault({})
      .AsExtra();
  AddAttr<std::string>(OpDeviceAttrName(), "Device type of this operator.")
      .SetDefault("")
      .AsExtra();

  AddAttr<bool>(OpWithQuantAttrName(),
                "Whether the operator has attributes used by quantization. ")
      .SetDefault(false)
      .AsExtra();
  // 检查 Input、Attr、Output 没有重名的
  Validate();
}

}  // namespace framework
}  // namespace paddle
