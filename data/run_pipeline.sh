#!/bin/bash
# Full data pipeline per mission.md Phase 0
# Run from project root: bash data/run_pipeline.sh
# All generated data goes to output/ (not data/, which is git-tracked)

set -e

echo "=== Step 0a: Collect from widget-factory (10K) ==="
python -m data.collection.collect_widget_factory --num_samples 10000 --output_dir ./output/raw --workers 8

echo "=== Step 0b: Generate synthetic widgets ==="
python -m data.collection.generate_synthetic --num_samples 10000 --output_dir ./output/raw --workers 8

echo "=== Step 1: Render verification (SSIM >= 0.85) ==="
python -m data.processing.render_verify --input_dir ./output/raw --output_dir ./output/verified --threshold 0.85

echo "=== Step 2: Deduplication (pHash) ==="
python -m data.processing.dedup --input_dir ./output/verified --output_dir ./output/deduped

echo "=== Step 3: Difficulty tagging ==="
python -m data.processing.tag_difficulty --input_dir ./output/deduped --output_dir ./output/tagged

echo "=== Step 4: CoT annotation ==="
python -m data.annotation.generate_cot --input_dir ./output/tagged --output_dir ./output/annotated --workers 8

echo "=== Step 5: Train/val/test split (90/5/5) ==="
python -m data.processing.split_data --input_dir ./output/annotated --output_dir ./output/final

echo "=== Pipeline complete! ==="
echo "Final dataset in ./output/final/{train,val,test}.json"
