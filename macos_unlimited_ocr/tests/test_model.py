from macos_unlimited_ocr.model import select_model_dtype


def test_select_model_dtype_uses_float32_for_mps():
    class FakeTorch:
        float16 = "float16"
        float32 = "float32"

    assert select_model_dtype(FakeTorch, "mps") == "float32"


def test_select_model_dtype_uses_float32_for_cpu():
    class FakeTorch:
        float16 = "float16"
        float32 = "float32"

    assert select_model_dtype(FakeTorch, "cpu") == "float32"
