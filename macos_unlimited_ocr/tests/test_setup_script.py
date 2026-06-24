from pathlib import Path


def test_setup_script_keeps_setuptools_below_torch_limit():
    script = Path("scripts/setup_venv.sh").read_text(encoding="utf-8")

    assert "setuptools<82" in script
    assert "pip install --upgrade pip setuptools wheel" not in script
