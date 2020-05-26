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

#include "Floor.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpFloor::filler(const tflite::Operator *, TFliteImport *, tflchef::ModelRecipe *) const
{
  // Nothing to do with filler
}

tflchef::Operation *TFliteOpFloor::build(const tflite::Operator *, TFliteImport *,
                                         tflchef::ModelRecipe *model_recipe) const
{
  auto operation = model_recipe->add_operation();

  operation->set_type("Floor");

  return operation;
}

} // namespace tflchef
