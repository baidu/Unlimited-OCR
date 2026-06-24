from __future__ import annotations

from contextlib import contextmanager, nullcontext


@contextmanager
def patch_remote_cuda_calls(torch_module, target_device: str):
    """Redirect remote-code CUDA assumptions for local macOS inference."""
    if target_device == "cuda":
        yield
        return

    tensor_type = torch_module.Tensor
    original_cuda = tensor_type.cuda
    original_to = tensor_type.to
    original_autocast = torch_module.autocast

    def redirected_cuda(self, *args, **kwargs):
        return self.to(target_device)

    def redirected_to(self, *args, **kwargs):
        if target_device != "cuda":
            bfloat16 = getattr(torch_module, "bfloat16", None)
            float32 = getattr(torch_module, "float32", None)
            if bfloat16 is not None and float32 is not None:
                args = tuple(float32 if arg is bfloat16 or arg == bfloat16 else arg for arg in args)
                if kwargs.get("dtype") is bfloat16 or kwargs.get("dtype") == bfloat16:
                    kwargs = {**kwargs, "dtype": float32}
        return original_to(self, *args, **kwargs)

    def redirected_autocast(device_type, *args, **kwargs):
        if device_type != "cuda":
            return original_autocast(device_type, *args, **kwargs)
        return nullcontext()

    tensor_type.cuda = redirected_cuda
    tensor_type.to = redirected_to
    torch_module.autocast = redirected_autocast
    try:
        yield
    finally:
        tensor_type.cuda = original_cuda
        tensor_type.to = original_to
        torch_module.autocast = original_autocast
