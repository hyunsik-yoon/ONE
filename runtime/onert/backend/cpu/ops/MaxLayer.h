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

#ifndef __ONERT_BACKEND_CPU_OPS_MAXLAYER_H__
#define __ONERT_BACKEND_CPU_OPS_MAXLAYER_H__

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

class MaxLayer : public ::onert::exec::IFunction
{
public:
  MaxLayer() : _lhs(nullptr), _rhs(nullptr), _output(nullptr)
  {
    // DO NOTHING
  }

public:
  template <typename T> void maximum();

  void maxQuant8();

  void configure(const IPortableTensor *lhs, const IPortableTensor *rhs, IPortableTensor *output);

  void run() override;

  const backend::ITensor *getOutput(int output_ind = 0) const override
  {
    assert(output_ind == 0);
    return _output;
  }

private:
  const IPortableTensor *_lhs;
  const IPortableTensor *_rhs;
  IPortableTensor *_output;
};

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_OPS_MAXLAYER_H__
