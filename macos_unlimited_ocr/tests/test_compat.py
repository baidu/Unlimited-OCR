from contextlib import nullcontext

from macos_unlimited_ocr.compat import patch_remote_cuda_calls


def test_patch_remote_cuda_calls_redirects_tensor_cuda():
    calls = []

    class FakeTensor:
        def cuda(self, *args, **kwargs):
            raise AssertionError("original cuda should not be called")

        def to(self, device):
            calls.append(device)
            return self

    class FakeTorch:
        Tensor = FakeTensor

        @staticmethod
        def autocast(device_type, *args, **kwargs):
            raise AssertionError("original cuda autocast should not be called")

    tensor = FakeTensor()
    with patch_remote_cuda_calls(FakeTorch, "cpu"):
        assert tensor.cuda() is tensor

    assert calls == ["cpu"]


def test_patch_remote_cuda_calls_maps_cuda_autocast_to_nullcontext_for_cpu():
    class FakeTensor:
        def cuda(self, *args, **kwargs):
            return self

        def to(self, device):
            return self

    class FakeTorch:
        Tensor = FakeTensor

        @staticmethod
        def autocast(device_type, *args, **kwargs):
            raise AssertionError("original cuda autocast should not be called")

    with patch_remote_cuda_calls(FakeTorch, "cpu"):
        ctx = FakeTorch.autocast("cuda")

    assert isinstance(ctx, nullcontext)


def test_patch_remote_cuda_calls_maps_cpu_bfloat16_tensor_to_float32():
    calls = []

    class FakeTensor:
        def cuda(self, *args, **kwargs):
            return self

        def to(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self

    class FakeTorch:
        Tensor = FakeTensor
        bfloat16 = "bfloat16"
        float32 = "float32"

        @staticmethod
        def autocast(device_type, *args, **kwargs):
            return nullcontext()

    tensor = FakeTensor()
    with patch_remote_cuda_calls(FakeTorch, "cpu"):
        tensor.to(FakeTorch.bfloat16)
        tensor.to(dtype=FakeTorch.bfloat16)

    assert calls == [(("float32",), {}), ((), {"dtype": "float32"})]


def test_patch_remote_cuda_calls_maps_mps_bfloat16_tensor_to_float32():
    calls = []

    class FakeTensor:
        def cuda(self, *args, **kwargs):
            return self

        def to(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self

    class FakeTorch:
        Tensor = FakeTensor
        bfloat16 = "bfloat16"
        float32 = "float32"

        @staticmethod
        def autocast(device_type, *args, **kwargs):
            raise AssertionError("cuda autocast should be disabled for mps compatibility")

    tensor = FakeTensor()
    with patch_remote_cuda_calls(FakeTorch, "mps"):
        tensor.to(FakeTorch.bfloat16)
        tensor.to(dtype=FakeTorch.bfloat16)
        ctx = FakeTorch.autocast("cuda", dtype=FakeTorch.bfloat16)

    assert calls == [(("float32",), {}), ((), {"dtype": "float32"})]
    assert isinstance(ctx, nullcontext)
