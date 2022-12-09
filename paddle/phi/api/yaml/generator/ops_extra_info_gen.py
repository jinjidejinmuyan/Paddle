# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import re

import yaml


def map_code_template(attrs_str, attrs_checker_str):
    return f"""// This file is generated by paddle/phi/api/yaml/generator/ops_extra_info_gen.py
#include "paddle/fluid/operators/ops_extra_info.h"

#include "paddle/phi/backends/gpu/cuda/cudnn_workspace_helper.h"

namespace paddle {{
namespace operators {{

ExtraInfoUtils::ExtraInfoUtils() {{
  g_extra_attrs_map_ = {{
    {attrs_str}
  }};

  g_extra_attrs_checker_ = {{
    {attrs_checker_str}
  }};
}}

}}  // namespace operators
}}  // namespace paddle
"""


ATTR_TYPE_STRING_MAP = {
    'bool': 'bool',
    'int': 'int',
    'int64_t': 'int64_t',
    'float': 'float',
    'double': 'double',
    'str': 'std::string',
    'int[]': 'std::vector<int>',
    'int64_t[]': 'std::vector<int64_t>',
    'float[]': 'std::vector<float>',
    'double[]': 'std::vector<double>',
    'str[]': 'std::vector<std::string>',
}


def parse_attr(attr_str):
    result = re.search(
        r"(?P<attr_type>[a-zA-Z0-9_[\]]+)\s+(?P<name>[a-zA-Z0-9_]+)\s*=\s*(?P<default_val>\S+)",
        attr_str,
    )
    return (
        ATTR_TYPE_STRING_MAP[result.group('attr_type')],
        result.group('name'),
        result.group('default_val'),
    )


def generate_extra_info(op_compat_yaml_path, ops_extra_info_path):
    compat_apis = []
    with open(op_compat_yaml_path, 'rt') as f:
        compat_apis = yaml.safe_load(f)

    def get_op_name(api_item):
        names = api_item.split('(')
        if len(names) == 1:
            return names[0].strip()
        else:
            return names[1].split(')')[0].strip()

    extra_map_str_list = []
    extra_checker_str_list = []

    for op_compat_args in compat_apis:
        if 'extra' in op_compat_args:
            extra_args_map = op_compat_args['extra']
            # TODO(chenweihang): add inputs and outputs
            if 'attrs' in extra_args_map:
                attr_map_list = []
                attr_checker_func_list = []
                for attr in extra_args_map['attrs']:
                    attr_type, attr_name, default_val = parse_attr(attr)
                    attr_checker_func_list.append(
                        f"[](framework::AttributeMap* attr_map, bool only_check_exist_value)-> void {{ ExtraAttrChecker<{attr_type}>(\"{attr_name}\", {default_val})(attr_map, only_check_exist_value);}}"
                    )
                    if attr_type.startswith("std::vector"):
                        attr_map_list.append(
                            f"{{\"{attr_name}\", {attr_type}{default_val}}}"
                        )
                    else:
                        attr_map_list.append(
                            f"{{\"{attr_name}\", {attr_type}{{{default_val}}}}}"
                        )
                api_extra_attr_map = ", ".join(attr_map_list)
                api_extra_attr_checkers = ",\n      ".join(
                    attr_checker_func_list
                )
                extra_map_str_list.append(
                    f"{{\"{get_op_name(op_compat_args['op'])}\", {{ {api_extra_attr_map} }}}}"
                )
                extra_checker_str_list.append(
                    f"{{\"{get_op_name(op_compat_args['op'])}\", {{ {api_extra_attr_checkers} }}}}"
                )
                if 'backward' in op_compat_args:
                    for bw_item in op_compat_args['backward'].split(','):
                        bw_op_name = get_op_name(bw_item)
                        extra_map_str_list.append(
                            f"{{\"{bw_op_name}\", {{ {api_extra_attr_map} }}}}"
                        )
                        extra_checker_str_list.append(
                            f"{{\"{bw_op_name}\", {{ {api_extra_attr_checkers} }}}}"
                        )

    ops_extra_info_file = open(ops_extra_info_path, 'w')
    ops_extra_info_file.write(
        map_code_template(
            ",\n    ".join(extra_map_str_list),
            ",\n    ".join(extra_checker_str_list),
        )
    )
    ops_extra_info_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle Extra Param Info for Op'
    )
    parser.add_argument(
        '--op_compat_yaml_path',
        help='path to api compat yaml file',
        default='paddle/phi/api/yaml/op_compat.yaml',
    )

    parser.add_argument(
        '--ops_extra_info_path',
        help='output of generated extra_prama_info code file',
        default='paddle/fluid/operators/ops_extra_info.cc',
    )

    options = parser.parse_args()

    op_compat_yaml_path = options.op_compat_yaml_path
    ops_extra_info_path = options.ops_extra_info_path

    generate_extra_info(op_compat_yaml_path, ops_extra_info_path)


if __name__ == '__main__':
    main()
