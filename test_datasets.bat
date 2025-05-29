@echo off
echo Starting datasets sanity check at %time% on %date%

python dataset_runner.py --dataset medmcqa --num-questions 3 --recruitment --recruitment-method intermediate --output-dir ./results/results_medmcqa

echo All evaluations completed at %time% on %date%