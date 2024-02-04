// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <string>

// BlockDiagonalCausalWithOffsetPaddedKeysMask
int _custom_mask_type(std::string bias_name) {
  if (bias_name == "LowerTriangularMask" ||
          bias_name == "BlockDiagonalCausalMask" ||
          bias_name == "BlockDiagonalCausalWithOffsetPaddedKeysMask")
   return 1;

  if (bias_name == "BlockDiagonalCausalFromBottomRightMask")
   return 2;
  return 0;
}