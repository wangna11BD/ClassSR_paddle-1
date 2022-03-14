# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import argparse
import paddle
from paddle.static import InputSpec
import options.options as option
import models.networks as networks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_inference_dir', default='./infer/', help='path where to save')
    parser.add_argument('-opt', type=str, default='../options/test/test_ClassSR_RCAN.yml', help='Path to options YMAL file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    test_model = networks.define_G(opt)
    
    test_model.eval()

    # decorate model with jit.save
    model = paddle.jit.to_static(
        test_model,
        input_spec=[
            InputSpec(shape=[-1, 3, -1, -1], dtype='float32', name='x'),
            InputSpec(shape=[-1, -1], dtype='int64', name='is_train')
        ])
    # save inference model
    paddle.jit.save(model, os.path.join('./infer/', "inference"))
    print(f"inference model has been saved into {opt.save_inference_dir}")

