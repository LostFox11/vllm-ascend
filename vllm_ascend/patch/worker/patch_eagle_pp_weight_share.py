#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from torch import nn

from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

_original_should_share = eagle_utils._should_share


def _has_weight(module: object) -> bool:
    return getattr(module, "weight", None) is not None


def _should_share(eagle: nn.Module, flag: str, draft, target) -> bool:
    """Avoid sharing EAGLE weights from PP placeholder layers.

    Under pipeline parallelism, layers that do not belong to the current rank
    can be represented by PPMissingLayer. It is still not None, but it has no
    weight attribute, so upstream's generic EAGLE sharing helper can fail while
    accessing target.weight. A missing-placeholder layer cannot be a valid
    sharing source, so keep the draft layer's own weights in that case.
    """
    if target is not None and not _has_weight(target):
        return False
    if draft is not None and not _has_weight(draft):
        return False
    return _original_should_share(eagle, flag, draft, target)


eagle_utils._should_share = _should_share
