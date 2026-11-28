# SPDX-License-Identifier: Apache-2.0
"""Host metadata for mask-free, single-query MLA FIA tasks."""

from bisect import bisect_right
from dataclasses import dataclass
from typing import NamedTuple

import torch

from .kv_compaction import CompactKV as CompactFIAKV
from .kv_compaction import compact_kv_rows as compact_fia_rows

__all__ = ["CompactFIAKV", "FIALengthProvider", "PackedC8KV", "compact_fia_rows"]


class PackedC8KV(NamedTuple):
    """Stable-packed BF16 [query, candidate, 576] KV with its invalid mask."""

    values: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True, eq=False)
class FIALengthProvider:
    # Each captured call owns a provider, including repeated calls in one layer.
    layer_name: str
    row_start: int
    rows: int
    capacity: int
    window: int | None
    ratio: int = 4
    budget: int = 511
    graph_queries: int | None = None

    def resolve(self, context):
        meta = context[self.layer_name] if isinstance(context, dict) else context
        lengths = []
        for row in range(self.row_start, self.row_start + self.rows):
            if self.graph_queries is not None:
                request, offset = divmod(row, self.graph_queries)
                count = self.graph_queries
            else:
                request = bisect_right(meta.query_start_loc, row) - 1
                offset = row - meta.query_start_loc[request]
                count = meta.query_start_loc[request + 1] - meta.query_start_loc[request]
            if request >= len(meta.seq_lens):
                # Padded graph requests use a zero KV row and are zeroed at output.
                lengths.append(1)
                continue
            visible = meta.seq_lens[request] - count + offset
            if self.window is not None:
                length = min(visible + 1, self.window)
            elif visible // self.ratio > self.budget:
                length = self.budget * self.ratio + visible % self.ratio + 1
            else:
                length = visible + 1
            lengths.append(max(1, min(length, self.capacity)))
        return {"actual_seq_kvlen": lengths}
