from typing import Tuple

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from bnn_pref.utils.type import QueryData, unpackable_dataclass


@unpackable_dataclass
class QueryBuffer:
    """
    Store all queries received so far, for sgd training
    """

    contexts: Float[Array, "Q 2 T D"]
    labels: Float[Array, "Q 2"]
    ptr: int = 0  # points to the next empty slot in the buffer
    max_size: int = 100

    def __len__(self) -> int:
        return self.ptr

    @classmethod
    def create(cls, max_size: int, traj_shape: Tuple[int, ...]):
        buffer = QueryBuffer(
            contexts=jnp.empty((max_size, 2, *traj_shape)),
            labels=jnp.empty((max_size, 2)),
            ptr=0,
            max_size=max_size,
        )
        return buffer

    def add_samples(self, new: QueryData):
        """Update the buffer with new query data."""
        n_new = new.contexts.shape[0]
        assert new.contexts.ndim == self.contexts.ndim, "contexts must have same ndim"
        assert new.labels.ndim == self.labels.ndim, "labels must have same ndim"
        assert self.ptr + n_new <= self.max_size, "buffer overflow"

        new_contexts = lax.dynamic_update_slice_in_dim(
            self.contexts, new.contexts, self.ptr, 0
        )
        new_labels = lax.dynamic_update_slice_in_dim(
            self.labels, new.labels, self.ptr, 0
        )
        new_ptr = self.ptr + n_new

        self = self.replace(
            contexts=new_contexts,
            labels=new_labels,
            ptr=new_ptr,
        )
        return self

    def get_all(self) -> QueryData:
        """Get all valid queries from the buffer."""
        contexts = lax.dynamic_slice_in_dim(self.contexts, 0, slice_size=self.ptr)
        labels = lax.dynamic_slice_in_dim(self.labels, 0, slice_size=self.ptr)
        data = QueryData(contexts=contexts, labels=labels)
        return data

    def get_newest_n(self, n: int) -> QueryData:
        """Get the most recent `n` queries from the buffer."""
        assert n <= self.ptr, "n must be less than or equal to the buffer size"
        start = self.ptr - n
        contexts = lax.dynamic_slice_in_dim(self.contexts, start, slice_size=n)
        labels = lax.dynamic_slice_in_dim(self.labels, start, slice_size=n)
        data = QueryData(contexts=contexts, labels=labels)
        return data
