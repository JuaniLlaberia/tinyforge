import torch

def detect_precision() -> tuple[torch.dtype, bool]:
    """
    Detects the training dtype for the current GPU.

    Returns:
        tuple[torch.dtype, bool]: (dtype, needs_grad_scaler).
            - No CUDA (CPU dev/smoke tests): (float32, False).
            - CUDA on A100/L4 (or any GPU reporting bf16 support): (bfloat16, False).
            - CUDA otherwise (e.g. T4, no bf16): (float16, True).
    """
    if not torch.cuda.is_available():
        return torch.float32, False

    device_name = torch.cuda.get_device_name(0)
    if "A100" in device_name or "L4" in device_name or torch.cuda.is_bf16_supported():
        return torch.bfloat16, False

    return torch.float16, True
