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
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medrag/medmcqa_specialsetINT --leadership --trust --medrag --seed 111
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medrag/medmcqa_specialsetINT --leadership --trust --medrag --seed 222
::python dataset_runner.py --dataset medmcqa --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medrag/medmcqa_specialsetINT --leadership --trust --medrag --seed 333

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

REM =============================================================================
REM DDXPlus Dataset - Special Set: Leadership + Closed-loop + Team Orientation + Mutual Trust
REM =============================================================================
::python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/ddxplus --leadership --closedloop --orientation --trust --seed 111 --n-max 2
::python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/ddxplus --leadership --closedloop --orientation --trust --seed 222 --n-max 2
::python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/ddxplus --leadership --closedloop --orientation --trust --seed 333 --n-max 2

REM =============================================================================
REM MedBullets Dataset - Special Set: Leadership + Closed-loop + Team Orientation + Mutual Trust
REM =============================================================================
::python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/medbullets --leadership --closedloop --orientation --trust --seed 111 --n-max 2
::python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/medbullets --leadership --closedloop --orientation --trust --seed 222 --n-max 2
::python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/medbullets --leadership --closedloop --orientation --trust --seed 333 --n-max 2

REM =============================================================================
REM SymCat Dataset - Special Set: Leadership + Closed-loop + Team Orientation + Mutual Trust
REM =============================================================================
::python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/symcat --leadership --closedloop --orientation --trust --seed 111 --n-max 3
::python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/symcat --leadership --closedloop --orientation --trust --seed 222 --n-max 3
::python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/special/symcat --leadership --closedloop --orientation --trust --seed 333 --n-max 3


@echo off
REM =============================================================================
REM COMPREHENSIVE MEDICAL DATASET EVALUATION SUITE
REM New Datasets: DDXPlus, MedBullets, SymCat
REM Configurations: Leadership, Closed-loop, Mutual Monitoring, Shared Mental Model, 
REM                Team Orientation, Mutual Trust, All Features
REM =============================================================================

echo Starting comprehensive evaluation of new medical datasets...
echo Datasets: DDXPlus, MedBullets, SymCat
echo Configurations: 7 different teamwork configurations
echo Seeds: 111, 222, 333 for reproducibility
echo.



REM =============================================================================
REM SYMCAT DATASET - All Features Configuration
REM =============================================================================

python dataset_runner.py --dataset symcat --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 2 --seed 111 --output-dir ./results/symcat_n2
python dataset_runner.py --dataset symcat --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 2 --seed 222 --output-dir ./results/symcat_n2
python dataset_runner.py --dataset symcat --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 2 --seed 333 --output-dir ./results/symcat_n2

python dataset_runner.py --dataset symcat --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 3 --seed 111 --output-dir ./results/symcat_n3
python dataset_runner.py --dataset symcat --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 3 --seed 222 --output-dir ./results/symcat_n3
python dataset_runner.py --dataset symcat --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 3 --seed 333 --output-dir ./results/symcat_n3

python dataset_runner.py --dataset symcat --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 4 --seed 111 --output-dir ./results/symcat_n4
python dataset_runner.py --dataset symcat --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 4 --seed 222 --output-dir ./results/symcat_n4
python dataset_runner.py --dataset symcat --num-questions 50 --all --recruitment --recruitment-method intermediate --n-max 4 --seed 333 --output-dir ./results/symcat_n4

REM =============================================================================
REM EVALUATION SUMMARY
REM =============================================================================
echo.
echo ===============================================================================
echo COMPREHENSIVE MEDICAL DATASET EVALUATION COMPLETED!
echo ===============================================================================
echo.
echo Summary:
echo - Datasets evaluated: DDXPlus, MedBullets, SymCat
echo - Configurations tested: 7 (Leadership, Closed-loop, Mutual Monitoring, 
echo                             Shared Mental Model, Team Orientation, 
echo                             Mutual Trust, All Features)
echo - Seeds used: 111, 222, 333 (for reproducibility)
echo - Questions per run: 50
echo - Total runs: 63 (3 datasets × 7 configurations × 3 seeds)
echo - Total questions evaluated: 3,150
echo.
echo Results saved in:
echo   ./results/ddxplus/[configuration]/
echo   ./results/medbullets/[configuration]/  
echo   ./results/symcat/[configuration]/
echo.
echo Next steps:
echo 1. Check results in the output directories
echo 2. Analyze performance across configurations
echo 3. Compare medical reasoning effectiveness
echo 4. Review agent recruitment patterns
echo.
echo Evaluation complete! Check the results directories for detailed analysis.

pause

::
::python dataset_runner.py --dataset pmc_vqa --num-questions 5 --recruitment --recruitment-method intermediate --n-max 4 --output-dir ./results/PMC_TEST
::python dataset_runner.py --dataset path_vqa --num-questions 5 --recruitment --recruitment-method intermediate --n-max 4 --output-dir ./results/PATH_TEST

echo "All special set experiments completed!"
pause

