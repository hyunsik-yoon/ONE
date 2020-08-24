/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_GRAPH_PASS_PERMUTATION_OPERATION_PASS_H__
#define __ONERT_GRAPH_PASS_PERMUTATION_OPERATION_PASS_H__

#include "ir/OperationVisitor.h"
#include "LoweredOperationPass.h"

namespace onert
{
namespace ir
{
namespace pass
{

class PermutationOperationPass : public LoweredOperationPass, public OperationVisitor
{
public:
  using LoweredOperationPass::LoweredOperationPass;

public:
  std::string id() final { return "PermutationOperationPass"; }

public:
  void callback(const OperationIndex &i, Operation &n) final;

public:
  void visit(const operation::BinaryArithmetic &) final;
  void visit(const operation::Comparison &) final;
  void visit(const operation::Concat &) final;
  void visit(const operation::ElementwiseBinary &) final;
  void visit(const operation::ElementwiseUnary &) final;
  void visit(const operation::Pack &) final;
  void visit(const operation::PReLU &) final;
  void visit(const operation::SquaredDifference &) final;
  void visit(const operation::Unpack &) final;
  void visit(const operation::FullyConnected &) final;
  void visit(const operation::Gather &) final;
  void visit(const operation::Reshape &) final;

private:
  void applyExpandRanks(const Operation &);
  void changeToKeepLayout(const Operation &);
};

} // namespace pass
} // namespace ir
} // namespace onert

#endif // __ONERT_GRAPH_PASS_PERMUTATION_OPERATION_PASS_H__
