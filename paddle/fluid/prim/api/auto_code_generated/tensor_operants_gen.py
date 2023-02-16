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

import yaml
from prim_base import EagerPrimAPI

eager_header_include = """// Generated by paddle/fluid/prim/api/auto_code_generated/tensor_operants_gen.py

#pragma once

#include "paddle/phi/api/include/operants_base.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/macros.h"

"""

eager_header_start = """
namespace paddle {

namespace prim {

using Tensor = paddle::experimental::Tensor;
using TensorOperantsBase = paddle::operants::TensorOperantsBase;

class EagerTensorOperants : public TensorOperantsBase {
 private:
  DISABLE_COPY_AND_ASSIGN(EagerTensorOperants);

 public:
  EagerTensorOperants() = default;

"""


eager_header_end = """};

}  // namespace prim
}  // namespace paddle

"""


eager_source_include = """// Generated by paddle/fluid/prim/api/auto_code_generated/tensor_operants_gen.py

#include "paddle/fluid/prim/utils/eager/eager_tensor_operants.h"

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"

"""


eager_source_start = """
namespace paddle {

namespace prim {

"""


eager_source_end = """
}  // namespace prim
}  // namespace paddle

"""


static_header_include = """// Generated by paddle/fluid/prim/api/auto_code_generated/tensor_operants_gen.py

#pragma once

#include "paddle/phi/api/include/operants_base.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/macros.h"

"""

static_header_start = """
namespace paddle {

namespace prim {

using Tensor = paddle::experimental::Tensor;
using TensorOperantsBase = paddle::operants::TensorOperantsBase;

class StaticTensorOperants : public TensorOperantsBase {
 private:
  DISABLE_COPY_AND_ASSIGN(StaticTensorOperants);

 public:
  StaticTensorOperants() = default;

"""


static_header_end = """};

}  // namespace prim
}  // namespace paddle

"""


static_source_include = """// Generated by paddle/fluid/prim/api/auto_code_generated/tensor_operants_gen.py

#include "paddle/fluid/prim/utils/static/static_tensor_operants.h"

#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"

"""


static_source_start = """
namespace paddle {

namespace prim {
using DescTensor = paddle::prim::DescTensor;

"""


static_source_end = """
}  // namespace prim
}  // namespace paddle

"""


def generate_tensor_operants_api(
    api_yaml_path,
    eager_header_path,
    eager_source_path,
    static_header_path,
    static_source_path,
    api_prim_path,
):
    apis = []

    for each_api_yaml in api_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                apis.extend(api_list)

    eager_header_file = open(eager_header_path, 'w')
    eager_source_file = open(eager_source_path, 'w')
    static_header_file = open(static_header_path, 'w')
    static_source_file = open(static_source_path, 'w')

    eager_header_file.write(eager_header_include)
    eager_header_file.write(eager_header_start)
    eager_source_file.write(eager_source_include)
    eager_source_file.write(eager_source_start)
    static_header_file.write(static_header_include)
    static_header_file.write(static_header_start)
    static_source_file.write(static_source_include)
    static_source_file.write(static_source_start)

    with open(api_prim_path, 'rt') as f:
        api_prims = yaml.safe_load(f)
        # white list temporarily
        api_prims = ('add', 'subtract', 'multiply', 'divide')

    for api in apis:
        eager_api = EagerPrimAPI(api, api_prims)
        if eager_api.is_prim_api:
            eager_header_file.write(
                eager_api.gene_tensor_operants_declaration()
            )
            eager_source_file.write(
                eager_api.gene_eager_tensor_operants_implementation()
            )
            static_header_file.write(
                eager_api.gene_tensor_operants_declaration()
            )
            static_source_file.write(
                eager_api.gene_static_tensor_operants_implementation()
            )

    eager_header_file.write(eager_header_end)
    eager_source_file.write(eager_source_end)
    static_header_file.write(static_header_end)
    static_source_file.write(static_source_end)

    eager_header_file.close()
    eager_source_file.close()
    static_header_file.close()
    static_source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files'
    )
    parser.add_argument(
        '--api_yaml_path',
        help='path to api yaml file',
        nargs='+',
        default=['paddle/phi/api/yaml/ops.yaml'],
    )

    parser.add_argument(
        '--eager_tensor_operants_header_path',
        help='output of generated eager_tensor_operants header code file',
        default='paddle/fluid/prim/utils/eager/eager_tensor_operants.h.tmp',
    )

    parser.add_argument(
        '--eager_tensor_operants_source_path',
        help='output of generated eager_tensor_operants source code file',
        default='paddle/fluid/prim/utils/eager/eager_tensor_operants.cc.tmp',
    )

    parser.add_argument(
        '--static_tensor_operants_header_path',
        help='output of generated eager_tensor_operants header code file',
        default='paddle/fluid/prim/utils/static/static_tensor_operants.h.tmp',
    )

    parser.add_argument(
        '--static_tensor_operants_source_path',
        help='output of generated eager_tensor_operants source code file',
        default='paddle/fluid/prim/utils/static/static_tensor_operants.cc.tmp',
    )

    parser.add_argument(
        '--api_prim_yaml_path',
        help='Primitive API list yaml file.',
        default='paddle/fluid/prim/api/auto_code_generated/api.yaml',
    )

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    api_prim_yaml_path = options.api_prim_yaml_path
    eager_tensor_operants_header_path = (
        options.eager_tensor_operants_header_path
    )
    eager_tensor_operants_source_path = (
        options.eager_tensor_operants_source_path
    )
    static_tensor_operants_header_path = (
        options.static_tensor_operants_header_path
    )
    static_tensor_operants_source_path = (
        options.static_tensor_operants_source_path
    )

    generate_tensor_operants_api(
        api_yaml_path,
        eager_tensor_operants_header_path,
        eager_tensor_operants_source_path,
        static_tensor_operants_header_path,
        static_tensor_operants_source_path,
        api_prim_yaml_path,
    )


if __name__ == '__main__':
    main()
