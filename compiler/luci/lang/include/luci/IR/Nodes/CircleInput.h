/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LUCI_IR_CIRCLEINPUT_H__
#define __LUCI_IR_CIRCLEINPUT_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

#include <loco/IR/DataTypeTraits.h>
#include <loco/IR/GraphInputIndex.h>

namespace luci
{

/**
 * @brief CircleNode used for Input of the Graph
 * @note  This will not be exported as a specific op
 */
class CircleInput final : public FixedArityNode<0, CircleNodeImpl<CircleOpcode::CIRCLEINPUT>>
{
public:
  void index(const loco::GraphInputIndex &index);
  loco::GraphInputIndex index(void) const;

  bool indexed(void) const { return _index != -1; }

private:
  int64_t _index{-1}; // Uninitialized
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEINPUT_H__
