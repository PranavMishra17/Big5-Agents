@echo off
echo Starting evaluation for results_n3_s200 at %time% on %date%


echo Starting evaluation for results_n2_s300 at %time% on %date%
python dataset_runner.py --dataset medqa --num-questions 30 --output-dir ./results_cn2 --n-max 2 --all --seed 333
echo Completed evaluation at %time%

echo Starting evaluation for results_n2_s300 at %time% on %date%
python dataset_runner.py --dataset medqa --num-questions 30 --output-dir ./results_cn3 --n-max 3 --all --seed 333
echo Completed evaluation at %time%

echo Starting evaluation for results_n2_s200 at %time% on %date%
python dataset_runner.py --dataset medqa --num-questions 30 --output-dir ./results_cn4 --n-max 4 --all --seed 111
echo Completed evaluation at %time%


echo Starting evaluation for results_n2_s200 at %time% on %date%
python dataset_runner.py --dataset medqa --num-questions 30 --output-dir ./results_cn4 --n-max 4 --all --seed 222
echo Completed evaluation at %time%

echo Starting evaluation for results_n2_s200 at %time% on %date%
python dataset_runner.py --dataset medqa --num-questions 30 --output-dir ./results_cn4 --n-max 4 --all --seed 333
echo Completed evaluation at %time%

echo All evaluations completed at %time% on %date%