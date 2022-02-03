import math
import torch
from torch import Tensor
from torch.nn import Linear


class KSparseDense(Linear):

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)

        self._k_sparse_idx = 1
        self._k_sparse_idx_warmup = 10
        self._k_sparse_idx_warmup_repeat = 1.0

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)

        output = self._k_sieve(output)

        return output

    def _k_sieve(self, data: Tensor) -> Tensor:
        k_sparse_idx = self.current_k_sparse_idx(full_len=self._output_len, target_k_idx=self._k_sparse_idx)

        values, indices = torch.topk(data, k=k_sparse_idx, sorted=True)
        values = torch.reshape(values, [-1])

        data_ndims = data.ndim
        data_shape = torch.tensor(list(data.shape))

        shapes = torch.unbind(data_shape[:(data.ndim - 1)]) + (torch.tensor(k_sparse_idx),)

        temp_indices = torch.meshgrid(*[torch.range(start=0, end=d.item()) for d in shapes], indexing='ij')

        temp_indices = torch.stack(temp_indices[:-1] + [indices], axis=-1)

        full_indices = torch.reshape(temp_indices, [-1, data_ndims])

        mask_st = tf.SparseTensor(indices=tf.cast(full_indices, dtype=tf.int64),
                                  values=tf.ones_like(values),
                                  dense_shape=tf.cast(data_shape, dtype=tf.int64))

        mask = tf.sparse_tensor_to_dense(tf.sparse_reorder(mask_st))



        sparse_data = torch.multiply(data, mask)

        return sparse_data

    def current_k_sparse_idx(self, full_len: int, target_k_idx: int) -> int:

        def linear_decrease():
            ideal = full_len - (full_len - target_k_idx) * warmup
            rounded = torch.tensor(round(ideal))
            clipped = torch.clamp(rounded, target_k_idx, full_len)
            return clipped

        warmup = self._k_sparse_idx_warmup
        current_k_idx = linear_decrease() if warmup < 1.0 else target_k_idx

        return current_k_idx

    def _op_k_sparse_idx_warmup(self, epoch):
        epoch = epoch - 1  # counter starts from 1, so it needs to be decreased

        # prepare updating
        epoch_quantized = torch.floor_divide(epoch, self._k_sparse_idx_warmup_repeat) * self._k_sparse_idx_warmup_repeat
        progress = 1.0 * epoch_quantized / self._k_sparse_idx_warmup
        clipped = torch.clamp(progress, 0.0, 1.0)

        # check if update is needed
        cond_whether_warmed_up = 1.0 if epoch < 0 else clipped

        return cond_whether_warmed_up
