import torch
import numpy as np


class GradientCompressionEngine:
    def __init__(self, compression_ratio=0.3):
        self.compression_ratio = compression_ratio
        self.error_accumulator = None

    def compress_topk(self, gradients):
        flat_grad = torch.cat([g.flatten() for g in gradients])
        
        num_elements = flat_grad.numel()
        k_elements = max(1, int(num_elements * self.compression_ratio))
        
        topk_values, topk_indices = torch.topk(torch.abs(flat_grad), k_elements)
        
        compressed_grad = torch.zeros_like(flat_grad)
        compressed_grad[topk_indices] = flat_grad[topk_indices]
        
        return compressed_grad, topk_indices

    def decompress_topk(self, compressed_grad, topk_indices, original_shape):
        decompressed = torch.zeros_like(compressed_grad)
        decompressed[topk_indices] = compressed_grad[topk_indices]
        
        return decompressed

    def compress_with_error_feedback(self, gradients, momentum_factor=0.9):
        flat_grad = torch.cat([g.flatten() for g in gradients])
        
        if self.error_accumulator is None:
            self.error_accumulator = torch.zeros_like(flat_grad)
        
        compensated_grad = flat_grad + self.error_accumulator
        
        num_elements = compensated_grad.numel()
        k_elements = max(1, int(num_elements * self.compression_ratio))
        
        topk_values, topk_indices = torch.topk(torch.abs(compensated_grad), k_elements)
        
        compressed_grad = torch.zeros_like(compensated_grad)
        compressed_grad[topk_indices] = compensated_grad[topk_indices]
        
        decompressed_grad = torch.zeros_like(compensated_grad)
        decompressed_grad[topk_indices] = compressed_grad[topk_indices]
        
        self.error_accumulator = compensated_grad - decompressed_grad
        
        return compressed_grad, topk_indices

    def quantize_gradients(self, gradients, num_bits=8):
        flat_grad = torch.cat([g.flatten() for g in gradients])
        
        min_val = torch.min(flat_grad)
        max_val = torch.max(flat_grad)
        
        scale = (max_val - min_val) / (2 ** num_bits - 1)
        
        if scale == 0:
            scale = 1.0
        
        quantized = torch.round((flat_grad - min_val) / scale)
        
        return quantized, min_val, scale

    def dequantize_gradients(self, quantized_grad, min_val, scale):
        dequantized = quantized_grad * scale + min_val
        return dequantized

    def compress_random_sparsification(self, gradients, sparsity_ratio=0.7):
        flat_grad = torch.cat([g.flatten() for g in gradients])
        
        num_elements = flat_grad.numel()
        num_keep = max(1, int(num_elements * (1 - sparsity_ratio)))
        
        indices = torch.randperm(num_elements)[:num_keep]
        
        compressed_grad = torch.zeros_like(flat_grad)
        compressed_grad[indices] = flat_grad[indices]
        
        scaling_factor = num_elements / num_keep
        compressed_grad = compressed_grad * scaling_factor
        
        return compressed_grad, indices

    def get_compression_ratio(self, original_size, compressed_size):
        if original_size == 0:
            return 0.0
        
        return 1.0 - (compressed_size / original_size)

    def reset_error_accumulator(self):
        self.error_accumulator = None

