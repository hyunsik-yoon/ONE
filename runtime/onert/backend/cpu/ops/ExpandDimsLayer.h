/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ExpandDi__ONERT_BACKEND_CPU_OPS_EXPANDDIMS_LAYER_H__ms
#define ExpandDi__ONERT_BACKEND_CPU_OPS_EXPANDDIMS_LAYER_H__ms

#include <backend/IPortableTensor.h>

#include <exec/IFunction.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

class ExpandDimsLayer : public ::onert::exec::IFunction
{
public:
  ExpandDimsLayer();

public:
  void configure(const IPortableTensor *input, const IPortableTensor *axis,
                 IPortableTensor *output);

  void run() override;

  const backend::ITensor *getOutput(int output_ind = 0) const override
  {
    assert(output_ind == 0);
    return _output;
  }

private:
  const IPortableTensor *_input;
  const IPortableTensor *_axis;
  IPortableTensor *_output;
};

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // ExpandDi__ONERT_BACKEND_CPU_OPS_EXPANDDIMS_LAYER_H__ms
