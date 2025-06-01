@echo off
echo Starting datasets sanity check at %time% on %date%

python dataset_runner.py --dataset medmcqa --num-questions 30 --recruitment --recruitment-method intermediate --output-dir ./results/results_medmcqa_3 --n-max 3

python dataset_runner.py --dataset pubmedqa --num-questions 30 --recruitment --recruitment-method intermediate --output-dir ./results/results_pubmedqa --n-max 3

python dataset_runner.py --dataset mmlupro-med --num-questions 30 --recruitment --recruitment-method intermediate --output-dir ./results/results_mmlu --n-max 3

echo All evaluations completed at %time% on %date%