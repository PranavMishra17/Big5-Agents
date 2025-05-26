@echo off
echo Starting datasets sanity check at %time% on %date%

python dataset_runner.py --dataset mmlupro-med --num-questions 5 --recruitment --output-dir ./results_mmlupro
python dataset_runner.py --dataset pubmedqa --num-questions 5 --recruitment --output-dir ./results_pubmedqa
python dataset_runner.py --dataset medmcqa --num-questions 5 --recruitment --output-dir ./results_medmcqa 

echo All evaluations completed at %time% on %date%