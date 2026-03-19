#!/bin/bash
# Setup Widget2Code benchmark evaluation
# Run from project root: bash evaluation/setup_benchmark.sh

set -e

echo "=== Cloning Widget2Code repo ==="
if [ ! -d "./evaluation/widget2code" ]; then
    git clone https://github.com/Djanghao/widget2code.git ./evaluation/widget2code
else
    echo "Already cloned"
fi

echo "=== Downloading benchmark dataset ==="
if [ ! -d "./data/widget2code-benchmark" ]; then
    pip install huggingface_hub
    huggingface-cli download Djanghao/widget2code-benchmark --repo-type dataset --local-dir ./data/widget2code-benchmark
else
    echo "Already downloaded"
fi

echo "=== Downloading benchmark results (for comparison) ==="
if [ ! -d "./evaluation/benchmarks_backup" ]; then
    pip install gdown
    gdown --fuzzy "https://drive.google.com/file/d/1LAYReu4fUES1IE0qM7h-zNGvyUgYnqwz/view?usp=sharing" -O ./evaluation/benchmarks_backup.zip
    unzip ./evaluation/benchmarks_backup.zip -d ./evaluation/benchmarks_backup
    rm ./evaluation/benchmarks_backup.zip
else
    echo "Already downloaded"
fi

echo "=== Setup complete ==="
echo "Ground truth: ./data/widget2code-benchmark/test"
echo ""
echo "To generate + evaluate:"
echo "  python -m evaluation.run_benchmarks --checkpoint /shared/advey/checkpoints/grpo_final"
echo "  cd evaluation/widget2code"
echo "  ./scripts/evaluation/run_evaluation.sh ../../output/benchmark_results -g ../../data/widget2code-benchmark/test --cuda -w 16"
