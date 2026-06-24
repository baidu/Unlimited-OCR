from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_receipt.png"

    image = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=32)
    small = ImageFont.load_default(size=24)

    lines = [
        "Unlimited OCR macOS smoke test",
        "",
        "Invoice No: MAC-2026-0623",
        "Date: 2026-06-23",
        "Item        Qty    Price",
        "Notebook     2     12.50",
        "Coffee       1      4.20",
        "Total              29.20",
    ]
    y = 40
    for i, line in enumerate(lines):
        draw.text((50, y), line, fill="black", font=font if i == 0 else small)
        y += 52

    image.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
