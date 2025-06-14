@echo off
echo Starting evaluation for MEDQA at %time% on %date%

::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n2 --n-max 2 --seed 111
::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n2 --n-max 2 --seed 222
::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n2 --n-max 2 --seed 333

::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n3 --n-max 3 --seed 111
::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n3 --n-max 3 --seed 222
::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n3 --n-max 3 --seed 333

::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n4 --n-max 4 --seed 111
::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n4 --n-max 4 --seed 222
::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n4 --n-max 4 --seed 333

::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n5 --n-max 5 --seed 111
::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n5 --n-max 5 --seed 222
::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medqa_n5 --n-max 5 --seed 333


echo All MEDQA EVAL completed at %time% on %date%
echo _________________________________________________________________________________________
echo              ##########################################################
echo _________________________________________________________________________________________
echo Starting evaluation for MEDMCQA at %time% on %date%


::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medmcqa_n2 --n-max 2 --seed 111
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medmcqa_n2 --n-max 2 --seed 222
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/fin/results_medmcqa_n2 --n-max 2 --seed 333

::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medmcqa_n3 --n-max 3 --seed 111
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medmcqa_n3 --n-max 3 --seed 222
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/fin/results_medmcqa_n3 --n-max 3 --seed 333

::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medmcqa_n4 --n-max 4 --seed 111
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_medmcqa_n4 --n-max 4 --seed 222
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/fin/results_medmcqa_n4 --n-max 4 --seed 333

::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/fin/results_medmcqa_n5 --n-max 5 --seed 111
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/fin/results_medmcqa_n5 --n-max 5 --seed 222
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/fin/results_medmcqa_n5 --n-max 5 --seed 333


echo All MEDMCQA EVAL completed at %time% on %date%
echo _________________________________________________________________________________________
echo              ##########################################################
echo _________________________________________________________________________________________

echo Starting evaluation for PubMedQA at %time% on %date%

::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_pubmedqa_n2 --n-max 2 --seed 111
::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_pubmedqa_n2 --n-max 2 --seed 222
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/fin/results_medmcqa_n2 --n-max 2 --seed 333

::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_pubmedqa_n3 --n-max 3 --seed 111
::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_pubmedqa_n3 --n-max 3 --seed 222
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/fin/results_medmcqa_n3 --n-max 3 --seed 333

::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_pubmedqa_n4 --n-max 4 --seed 111
::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_pubmedqa_n4 --n-max 4 --seed 222
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/fin/results_medmcqa_n4 --n-max 4 --seed 333

echo All MEDMCQA EVAL completed at %time% on %date%
echo _________________________________________________________________________________________
echo              ##########################################################
echo _________________________________________________________________________________________

echo Starting evaluation for MMLU-pro at %time% on %date%

python dataset_runner.py --dataset mmlupro-med --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_mmlupro-med_n2/111 --n-max 2 --seed 111
python dataset_runner.py --dataset mmlupro-med --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_mmlupro-med_n2/222 --n-max 2 --seed 222
python dataset_runner.py --dataset mmlupro-med --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_mmlupro-med_n2/333 --n-max 2 --seed 333

python dataset_runner.py --dataset mmlupro-med --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_mmlupro-med_n3/111 --n-max 3 --seed 111
python dataset_runner.py --dataset mmlupro-med --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_mmlupro-med_n3/222 --n-max 3 --seed 222
python dataset_runner.py --dataset mmlupro-med --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_mmlupro-med_n3/333 --n-max 3 --seed 333

python dataset_runner.py --dataset mmlupro-med --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_mmlupro-med_n4/111 --n-max 4 --seed 111
python dataset_runner.py --dataset mmlupro-med --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_mmlupro-med_n4/222 --n-max 4 --seed 222
python dataset_runner.py --dataset mmlupro-med --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_mmlupro-med_n4/333 --n-max 4 --seed 333

echo All mmlupro-med EVAL completed at %time% on %date%

echo All MEDMCQA EVAL completed at %time% on %date%
echo _________________________________________________________________________________________
echo              ##########################################################
echo _________________________________________________________________________________________

echo Starting evaluation for PubMedQA at %time% on %date%

::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_pubmedqa_n2 --n-max 2 --seed 111
::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_pubmedqa_n2 --n-max 2 --seed 222
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/fin/results_medmcqa_n2 --n-max 2 --seed 333

::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_pubmedqa_n3 --n-max 3 --seed 111
::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_pubmedqa_n3 --n-max 3 --seed 222
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/fin/results_medmcqa_n3 --n-max 3 --seed 333

::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_pubmedqa_n4 --n-max 4 --seed 111
::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --all --output-dir ./results/fin/results_pubmedqa_n4 --n-max 4 --seed 222
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/fin/results_medmcqa_n4 --n-max 4 --seed 333

echo All MEDMCQA EVAL completed at %time% on %date%
echo _________________________________________________________________________________________
echo              ##########################################################
echo _________________________________________________________________________________________
