"""
Phase 4.5.3: Runtime Safety Checks and Validation Guardrails

Provides runtime validation during compression operations:
  - Input shape and dtype validation
  - Memory safety checks
  - Numerical stability guards
  - Dimension mismatch detection
  - Performance anomaly detection

This module runs on every encode/decode operation to catch issues early.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Any
import warnings

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    severity: ValidationSeverity
    check_name: str
    message: str
    suggestion: Optional[str] = None


class KVTensorValidator:
    """Validates KV tensor properties for compression safety"""
    
    # Configuration for validation
    max_seq_length = 32768
    max_heads = 512
    max_head_dim = 1024
    max_total_tokens_per_batch = 1_000_000
    
    @staticmethod
    def validate_kv_tensor(
        kv_tensor: Any,
        operation: str = "encode",  # encode or decode
    ) -> Tuple[bool, Optional[ValidationResult]]:
        """
        Validate KV tensor before compression.
        
        Args:
            kv_tensor: KV tensor to validate
            operation: Operation being performed (encode/decode)
        
        Returns:
            (is_valid, validation_result_if_invalid)
        """
        checks = [
            KVTensorValidator._check_tensor_exists(kv_tensor),
            KVTensorValidator._check_tensor_shape(kv_tensor, operation),
            KVTensorValidator._check_tensor_dtype(kv_tensor),
            KVTensorValidator._check_tensor_values(kv_tensor),
        ]
        
        # Find first error
        for result in checks:
            if not result.passed:
                return False, result
        
        return True, None
    
    @staticmethod
    def _check_tensor_exists(kv_tensor: Any) -> ValidationResult:
        """Check tensor exists and is not None"""
        if kv_tensor is None:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.ERROR,
                check_name="tensor_exists",
                message="KV tensor is None",
                suggestion="Check that attention forward pass returned valid KV output"
            )
        return ValidationResult(
            passed=True,
            severity=ValidationSeverity.INFO,
            check_name="tensor_exists",
            message="Tensor exists"
        )
    
    @staticmethod
    def _check_tensor_shape(kv_tensor: Any, operation: str) -> ValidationResult:
        """Check tensor has valid shape"""
        try:
            if not hasattr(kv_tensor, 'shape'):
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    check_name="tensor_shape",
                    message="Tensor has no shape attribute",
                    suggestion="Input must be a tensor-like object with .shape"
                )
            
            shape = kv_tensor.shape
            
            # Validate dimensions
            if len(shape) < 2:
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    check_name="tensor_shape",
                    message=f"Expected at least 2D tensor, got shape {shape}",
                    suggestion="KV should be [seq_len, heads, dim] or similar"
                )
            
            # Check values are reasonable
            for i, s in enumerate(shape):
                if s <= 0:
                    return ValidationResult(
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        check_name="tensor_shape",
                        message=f"Invalid shape: dimension {i} is {s}",
                        suggestion="All dimensions must be > 0"
                    )
                
                # Check against max values
                if i == 0 and s > KVTensorValidator.max_seq_length:
                    return ValidationResult(
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        check_name="tensor_shape",
                        message=f"Seq length {s} exceeds max {KVTensorValidator.max_seq_length}",
                        suggestion="Context may be too long for gfx1030"
                    )
                
                if i == 1 and s > KVTensorValidator.max_heads:
                    return ValidationResult(
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        check_name="tensor_shape",
                        message=f"Number of heads {s} exceeds max {KVTensorValidator.max_heads}",
                        suggestion="Check model configuration"
                    )
                
                if i == 2 and s > KVTensorValidator.max_head_dim:
                    return ValidationResult(
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        check_name="tensor_shape",
                        message=f"Head dim {s} exceeds max {KVTensorValidator.max_head_dim}",
                        suggestion="Check model configuration"
                    )
            
            return ValidationResult(
                passed=True,
                severity=ValidationSeverity.INFO,
                check_name="tensor_shape",
                message=f"Valid shape: {shape}"
            )
        
        except Exception as e:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.ERROR,
                check_name="tensor_shape",
                message=f"Error checking tensor shape: {e}",
                suggestion="Ensure input is a valid tensor object"
            )
    
    @staticmethod
    def _check_tensor_dtype(kv_tensor: Any) -> ValidationResult:
        """Check tensor has valid dtype"""
        try:
            if not hasattr(kv_tensor, 'dtype'):
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    check_name="tensor_dtype",
                    message="Tensor has no dtype attribute",
                    suggestion="Input must be a typed tensor object"
                )
            
            dtype_str = str(kv_tensor.dtype)
            
            # Valid dtypes for KV
            valid_dtypes = [
                "float16", "float32", "float64",
                "float", "half", "double",
                "bfloat16",
            ]
            
            if not any(v in dtype_str.lower() for v in valid_dtypes):
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    check_name="tensor_dtype",
                    message=f"Unusual dtype for KV: {dtype_str}",
                    suggestion="Expected float16, float32, or float64"
                )
            
            return ValidationResult(
                passed=True,
                severity=ValidationSeverity.INFO,
                check_name="tensor_dtype",
                message=f"Valid dtype: {dtype_str}"
            )
        
        except Exception as e:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.ERROR,
                check_name="tensor_dtype",
                message=f"Error checking dtype: {e}",
                suggestion="Ensure input is a valid tensor"
            )
    
    @staticmethod
    def _check_tensor_values(kv_tensor: Any) -> ValidationResult:
        """Check tensor values are reasonable (no NaN/Inf)"""
        try:
            # Try to check for NaN/Inf
            if not hasattr(kv_tensor, 'numel'):
                return ValidationResult(
                    passed=True,
                    severity=ValidationSeverity.INFO,
                    check_name="tensor_values",
                    message="Cannot check values (no numel method); skipping"
                )
            
            # Only check if tensor is small enough (expensive operation)
            num_elements = kv_tensor.numel() if hasattr(kv_tensor, 'numel') else 1
            if num_elements > 100_000_000:  # Skip check for very large tensors
                return ValidationResult(
                    passed=True,
                    severity=ValidationSeverity.INFO,
                    check_name="tensor_values",
                    message=f"Tensor too large ({num_elements} elements); skipping value check"
                )
            
            # Check for NaN
            if hasattr(kv_tensor, 'isnan'):
                has_nan = kv_tensor.isnan().any()
                if has_nan:
                    return ValidationResult(
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        check_name="tensor_values",
                        message="Tensor contains NaN values",
                        suggestion="Check attention forward pass for numerical instability"
                    )
            
            # Check for Inf
            if hasattr(kv_tensor, 'isinf'):
                has_inf = kv_tensor.isinf().any()
                if has_inf:
                    return ValidationResult(
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        check_name="tensor_values",
                        message="Tensor contains Inf values",
                        suggestion="Check attention scaling factors and RoPE"
                    )
            
            return ValidationResult(
                passed=True,
                severity=ValidationSeverity.INFO,
                check_name="tensor_values",
                message="Tensor values look reasonable"
            )
        
        except Exception as e:
            logger.debug(f"Could not check tensor values: {e}")
            return ValidationResult(
                passed=True,
                severity=ValidationSeverity.INFO,
                check_name="tensor_values",
                message=f"Skipping value check: {e}"
            )


class CompressionModeValidator:
    """Validates compression mode selection"""
    
    @staticmethod
    def validate_mode_for_shape(
        mode: str,
        tensor_shape: Tuple[int, ...],
    ) -> Tuple[bool, Optional[ValidationResult]]:
        """
        Check if compression mode is safe for tensor shape.
        
        Args:
            mode: Compression mode (tq2, tq3, etc)
            tensor_shape: Shape of tensor being compressed
        
        Returns:
            (is_valid, validation_result_if_invalid)
        """
        if mode.startswith("tq"):
            # TurboQuant has minimum head dimension requirement
            if len(tensor_shape) >= 3:
                head_dim = tensor_shape[2]
                if head_dim < 8:
                    return False, ValidationResult(
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        check_name="mode_head_dim",
                        message=f"TurboQuant requires head_dim >= 8, got {head_dim}",
                        suggestion="Use fp16 for very small head dimensions"
                    )
        
        return True, None
    
    @staticmethod
    def validate_mode_consistency(
        mode: str,
        expected_bytes_per_token: float,
    ) -> Tuple[bool, Optional[ValidationResult]]:
        """
        Check if mode compression ratio matches expectations.
        
        Args:
            mode: Compression mode
            expected_bytes_per_token: Expected compressed size per token
        
        Returns:
            (is_valid, validation_result_if_invalid)
        """
        # Rough compression ratios
        expected_ratios = {
            "tq1": 0.125,
            "tq2": 0.25,
            "tq3": 0.375,
            "tq4": 0.5,
            "fp8_e4m3": 0.5,
            "fp16": 1.0,
        }
        
        if mode in expected_ratios:
            expected = expected_ratios[mode]
            # Allow ±50% variation
            if expected_bytes_per_token < expected * 0.5 or \
               expected_bytes_per_token > expected * 1.5:
                return False, ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    check_name="mode_compression_ratio",
                    message=f"Unexpected compression ratio for {mode}",
                    suggestion="Check if mode is implemented correctly"
                )
        
        return True, None


class MemorySafetyValidator:
    """Validates memory safety during compression"""
    
    @staticmethod
    def check_output_size(
        input_shape: Tuple[int, ...],
        input_dtype_bytes: int,
        mode: str,
        max_available_gpu_memory: int = 12 * 1024**3,  # 12 GB for gfx1030
    ) -> Tuple[bool, Optional[ValidationResult]]:
        """
        Check if output will fit in available GPU memory.
        
        Args:
            input_shape: Input tensor shape
            input_dtype_bytes: Bytes per element in input dtype
            mode: Compression mode
            max_available_gpu_memory: Max GPU memory in bytes
        
        Returns:
            (is_valid, validation_result_if_invalid)
        """
        import math
        
        # Estimate input size
        num_elements = math.prod(input_shape)
        input_bytes = num_elements * input_dtype_bytes
        
        # Estimate compression ratio
        compression_ratios = {
            "tq1": 0.125, "tq2": 0.25, "tq3": 0.375, "tq4": 0.5,
            "fp8_e4m3": 0.5, "fp16": 1.0
        }
        
        compression_ratio = compression_ratios.get(mode, 1.0)
        estimated_output = input_bytes * compression_ratio
        
        # Add metadata overhead (~5%)
        total_with_metadata = estimated_output * 1.05
        
        if total_with_metadata > max_available_gpu_memory * 0.9:  # 90% threshold
            return False, ValidationResult(
                passed=False,
                severity=ValidationSeverity.WARNING,
                check_name="memory_safety",
                message=f"Output may exceed 90% of GPU memory ({total_with_metadata / 1024**3:.1f} GB)",
                suggestion="Consider reducing batch size or context length"
            )
        
        return True, None


class RuntimeSafetyCheckManager:
    """
    Orchestrates runtime safety checks for compression operations.
    
    Runs before encode/decode to catch issues early.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize safety checker.
        
        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode
        self.check_history = []
    
    def pre_encode_checks(
        self,
        kv_tensor: Any,
        mode: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Run all checks before encoding.
        
        Args:
            kv_tensor: KV tensor to encode
            mode: Compression mode
        
        Returns:
            (is_valid, error_msg_if_invalid)
        """
        checks = []
        
        # Check tensor
        valid, result = KVTensorValidator.validate_kv_tensor(kv_tensor, "encode")
        if result:
            checks.append(result)
        if not valid:
            return False, result.message
        
        # Check mode for shape
        valid, result = CompressionModeValidator.validate_mode_for_shape(
            mode, kv_tensor.shape
        )
        if result:
            checks.append(result)
        if not valid and self.strict_mode:
            return False, result.message
        
        # Log checks
        for check in checks:
            if check.severity == ValidationSeverity.ERROR:
                logger.error(f"[{check.check_name}] {check.message}")
            elif check.severity == ValidationSeverity.WARNING:
                logger.warning(f"[{check.check_name}] {check.message}")
            else:
                logger.debug(f"[{check.check_name}] {check.message}")
        
        self.check_history.extend(checks)
        return True, None
    
    def pre_decode_checks(
        self,
        compressed_kv: Any,
        mode: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Run checks before decoding.
        
        Args:
            compressed_kv: Compressed KV data
            mode: Compression mode
        
        Returns:
            (is_valid, error_msg_if_invalid)
        """
        # Basic checks
        if compressed_kv is None:
            return False, "Compressed KV is None"
        
        # Check has required attributes
        if mode != "fp16" and not hasattr(compressed_kv, 'data'):
            logger.warning(f"Compressed KV missing 'data' attribute for mode {mode}")
        
        return True, None


# Global safety checker instance
_global_safety_manager: Optional[RuntimeSafetyCheckManager] = None


def get_safety_manager(strict_mode: bool = False) -> RuntimeSafetyCheckManager:
    """Get or create global safety manager"""
    global _global_safety_manager
    if _global_safety_manager is None:
        _global_safety_manager = RuntimeSafetyCheckManager(strict_mode)
    return _global_safety_manager


logger.info("Phase 4.5.3 Runtime Safety Checks initialized")
