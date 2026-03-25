"""Re-extract TSX from raw outputs and render to PNG.

Usage:
    python -m evaluation.rerender /shared/advey/hf-benchmark/sft
    python -m evaluation.rerender /shared/advey/hf-benchmark/base
"""

import glob
import os
import sys
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.run_sft_benchmark import extract_code, _cleanup_tsx
from reward.programmatic import render_tsx_to_image


def _get_image_size(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        header = f.read(24)
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", header[16:24])
    return (800, 600)


def main():
    output_dir = sys.argv[1]
    gt_dir = sys.argv[2] if len(sys.argv) > 2 else "/shared/advey/hf-benchmark/gt"

    raw_files = sorted(glob.glob(os.path.join(output_dir, "*/raw_output.txt")))
    print(f"Found {len(raw_files)} raw outputs in {output_dir}")

    re_extracted = 0
    rendered = 0
    render_errors = 0
    skipped = 0

    for i, raw_path in enumerate(raw_files):
        widget_id = os.path.basename(os.path.dirname(raw_path))
        sample_dir = os.path.dirname(raw_path)
        tsx_path = os.path.join(sample_dir, "component.tsx")
        pred_path = os.path.join(sample_dir, "pred.png")
        gt_path = os.path.join(gt_dir, f"gt_{widget_id}.png")

        # Skip if already rendered
        if os.path.exists(pred_path) and os.path.getsize(pred_path) > 5000:
            skipped += 1
            continue

        # Re-extract TSX
        raw_text = open(raw_path).read()
        tsx = extract_code(raw_text)
        if tsx is None:
            continue

        # Write TSX
        with open(tsx_path, "w") as f:
            f.write(tsx)
        re_extracted += 1

        # Render
        try:
            w, h = _get_image_size(gt_path) if os.path.exists(gt_path) else (800, 600)
            rendered_bytes = render_tsx_to_image(tsx, w, h)
            with open(pred_path, "wb") as f:
                f.write(rendered_bytes)
            rendered += 1
        except Exception as e:
            render_errors += 1
            print(f"  [{widget_id}] Render error: {e}")

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(raw_files)} ({rendered} rendered, {render_errors} errors, {skipped} skipped)")

    print(f"\nDone: {re_extracted} re-extracted, {rendered} rendered, {render_errors} errors, {skipped} skipped")


if __name__ == "__main__":
    main()
