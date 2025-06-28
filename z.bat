::python dataset_runner.py --dataset symcat --num-questions 10 --recruitment --recruitment-method intermediate --n-max 5 --seed 111 --output-dir ./results/symcat

::python dataset_runner.py --dataset pmc_vqa --num-questions 5 --recruitment --recruitment-method intermediate --n-max 4 --output-dir ./results/PMC_TEST

::python dataset_runner.py --dataset path_vqa --num-questions 5 --recruitment --recruitment-method intermediate --n-max 4 --output-dir ./results/PATH_TEST

::# DDXPlus dataset

python dataset_runner.py --dataset ddxplus --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 4 --seed 111 --output-dir ./results/ddxplus_n4
python dataset_runner.py --dataset ddxplus --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 4 --seed 222 --output-dir ./results/ddxplus_n4
python dataset_runner.py --dataset ddxplus --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 4 --seed 333 --output-dir ./results/ddxplus_n4

::# MedBullets dataset

python dataset_runner.py --dataset medbullets --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 4 --seed 111 --output-dir ./results/medbullets_n4
python dataset_runner.py --dataset medbullets --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 4 --seed 222 --output-dir ./results/medbullets_n4
python dataset_runner.py --dataset medbullets --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 4 --seed 333 --output-dir ./results/medbullets_n4