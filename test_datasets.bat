@echo off
echo Starting datasets sanity check at %time% on %date%

python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --output-dir ./results_medmcqa3

echo All evaluations completed at %time% on %date%