/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Service/CircleNodeClone.h"

#include <gtest/gtest.h>

TEST(CloneNodeTest, clone_FullyConnected)
{
  auto g = loco::make_graph();
  auto node_fc = g->nodes()->create<luci::CircleFullyConnected>();
  node_fc->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_fc->weights_format(luci::CircleFullyConnected::WeightsFormat::DEFAULT);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_fc, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_fc = dynamic_cast<luci::CircleFullyConnected *>(cloned);
  ASSERT_NE(nullptr, cloned_fc);
  ASSERT_EQ(node_fc->fusedActivationFunction(), cloned_fc->fusedActivationFunction());
  ASSERT_EQ(node_fc->weights_format(), cloned_fc->weights_format());
}

TEST(CloneNodeTest, clone_FullyConnected_fusedact_NEG)
{
  auto g = loco::make_graph();
  auto node_fc = g->nodes()->create<luci::CircleFullyConnected>();
  node_fc->fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  node_fc->weights_format(luci::CircleFullyConnected::WeightsFormat::DEFAULT);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_fc, gc.get());
  ASSERT_EQ(nullptr, cloned);
}

TEST(CloneNodeTest, clone_FullyConnected_wf_NEG)
{
  auto g = loco::make_graph();
  auto node_fc = g->nodes()->create<luci::CircleFullyConnected>();
  node_fc->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_fc->weights_format(luci::CircleFullyConnected::WeightsFormat::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_fc, gc.get());
  ASSERT_EQ(nullptr, cloned);
}
