@echo off
echo Starting datasets sanity check at %time% on %date%

python dataset_runner.py --dataset medqa --num-questions 10 --recruitment --recruitment-method intermediate --output-dir ./results/test/parallel --n-max 5 --seed 111

::python dataset_runner.py --dataset mmlupro-med --num-questions 20 --recruitment --recruitment-method intermediate --output-dir ./results/test --n-max 5 --seed 444

::python dataset_runner.py --dataset pubmedqa --num-questions 25 --recruitment --recruitment-method intermediate --output-dir ./results/results_pubmedqa_c2 --n-max 3 --seed 222

::python dataset_runner.py --dataset mmlupro-med --num-questions 25 --recruitment --recruitment-method intermediate --output-dir ./results/results_mmlu_c2 --n-max 3 --seed 222

echo All evaluations completed at %time% on %date%