@echo off
echo Starting datasets sanity check at %time% on %date%

python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n2 --n-max 2 --seed 111
python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n2 --n-max 2 --seed 222
python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n2 --n-max 2 --seed 333

python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n3 --n-max 3 --seed 111
python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n3 --n-max 3 --seed 222
python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n3 --n-max 3 --seed 333

python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n4 --n-max 4 --seed 111
python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n4 --n-max 4 --seed 222
python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n4 --n-max 4 --seed 333

python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n5 --n-max 5 --seed 111
python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n5 --n-max 5 --seed 222
python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n5 --n-max 5 --seed 333



::python dataset_runner.py --dataset medmcqa --num-questions 25 --recruitment --recruitment-method intermediate --output-dir ./results/results_medmcqa_c2 --n-max 3 --seed 222

::python dataset_runner.py --dataset pubmedqa --num-questions 25 --recruitment --recruitment-method intermediate --output-dir ./results/results_pubmedqa_c2 --n-max 3 --seed 222

::python dataset_runner.py --dataset mmlupro-med --num-questions 25 --recruitment --recruitment-method intermediate --output-dir ./results/results_mmlu_c2 --n-max 3 --seed 222

echo All evaluations completed at %time% on %date%