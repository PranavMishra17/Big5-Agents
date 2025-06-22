@echo off
REM Special Sets Commands - Outperforming components for each dataset

REM =============================================================================
REM MedQA Dataset - Special Set: Closed-loop + Mutual Trust + Team Orientation
REM =============================================================================
::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method advanced --output-dir ./results/special/medqa_specialset --closedloop --trust --orientation --seed 111
::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method advanced --output-dir ./results/special/medqa_specialset --closedloop --trust --orientation --seed 222
::python dataset_runner.py --dataset medqa --num-questions 50 --recruitment --recruitment-method advanced --output-dir ./results/special/medqa_specialset --closedloop --trust --orientation --seed 333

REM =============================================================================
REM MedMCQA Dataset - Special Set: Leadership + Mutual Trust
REM =============================================================================
python dataset_runner.py --dataset medmcqa --num-questions 10 --recruitment --recruitment-method intermediate --output-dir ./results/medrag/medmcqa_specialsetINT --leadership --trust --medrag --seed 111
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/medmcqa_specialsetINT --leadership --trust --seed 222
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/medmcqa_specialsetINT --leadership --trust --seed 333

REM =============================================================================
REM PubMedQA Dataset - Special Set: Leadership + Closed-loop + Team Orientation + Mutual Trust
REM =============================================================================
::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/pubmedqa_specialsetINT --leadership --closedloop --orientation --trust --seed 111
::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/pubmedqa_specialsetINT --leadership --closedloop --orientation --trust --seed 222
::python dataset_runner.py --dataset pubmedqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/pubmedqa_specialsetINT --leadership --closedloop --orientation --trust --seed 333

REM =============================================================================
REM MMLU-Pro Dataset - Special Set: Leadership + Mutual Monitoring + Shared Mental Model + Team Orientation + Mutual Trust
REM =============================================================================
::python dataset_runner.py --dataset mmlupro-med --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/mmlupro_specialsetINT --leadership --mutual --mental --orientation --trust --seed 111
::python dataset_runner.py --dataset mmlupro-med --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/mmlupro_specialsetINT --leadership --mutual --mental --orientation --trust --seed 222
::python dataset_runner.py --dataset mmlupro-med --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/mmlupro_specialsetINT --leadership --mutual --mental --orientation --trust --seed 333

echo "All special set experiments completed!"
pause