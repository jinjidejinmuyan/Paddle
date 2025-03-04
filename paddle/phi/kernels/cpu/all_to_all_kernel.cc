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

#include "paddle/phi/kernels/all_to_all_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void AllToAllKernel(const Context& dev_ctx UNUSED,
                    const DenseTensor& x UNUSED,
                    DenseTensor* out UNUSED) {
  PADDLE_THROW(
      errors::Unimplemented("Unimplemented cpu kernel for all_to_all."));
}

}  // namespace phi

PD_REGISTER_KERNEL(all_to_all,
                   CPU,
                   ALL_LAYOUT,
                   phi::AllToAllKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::float16) {}
