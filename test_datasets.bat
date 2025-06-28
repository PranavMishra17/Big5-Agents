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
REM DDXPLUS DATASET - Leadership Configuration
REM =============================================================================
echo [DDXPlus] Running Leadership configuration...
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/leadership --leadership --seed 111 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/leadership --leadership --seed 222 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/leadership --leadership --seed 333 --n-max 2

REM =============================================================================
REM DDXPLUS DATASET - Closed-loop Configuration
REM =============================================================================
echo [DDXPlus] Running Closed-loop configuration...
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/closedloop --closedloop --seed 111 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/closedloop --closedloop --seed 222 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/closedloop --closedloop --seed 333 --n-max 2

REM =============================================================================
REM DDXPLUS DATASET - Mutual Monitoring Configuration
REM =============================================================================
echo [DDXPlus] Running Mutual Monitoring configuration...
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/mutual_monitoring --mutual --seed 111 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/mutual_monitoring --mutual --seed 222 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/mutual_monitoring --mutual --seed 333 --n-max 2

REM =============================================================================
REM DDXPLUS DATASET - Shared Mental Model Configuration
REM =============================================================================
echo [DDXPlus] Running Shared Mental Model configuration...
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/shared_mental_model --mental --seed 111 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/shared_mental_model --mental --seed 222 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/shared_mental_model --mental --seed 333 --n-max 2

REM =============================================================================
REM DDXPLUS DATASET - Team Orientation Configuration
REM =============================================================================
echo [DDXPlus] Running Team Orientation configuration...
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/team_orientation --orientation --seed 111 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/team_orientation --orientation --seed 222 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/team_orientation --orientation --seed 333 --n-max 2

REM =============================================================================
REM DDXPLUS DATASET - Mutual Trust Configuration
REM =============================================================================
echo [DDXPlus] Running Mutual Trust configuration...
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/mutual_trust --trust --seed 111 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/mutual_trust --trust --seed 222 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/mutual_trust --trust --seed 333 --n-max 2

REM =============================================================================
REM DDXPLUS DATASET - All Features Configuration
REM =============================================================================
echo [DDXPlus] Running All Features configuration...
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/all_features --leadership --closedloop --mutual --mental --orientation --trust --seed 111 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/all_features --leadership --closedloop --mutual --mental --orientation --trust --seed 222 --n-max 2
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/ddxplus2/all_features --leadership --closedloop --mutual --mental --orientation --trust --seed 333 --n-max 2

echo.
echo ===============================================================================
echo DDXPlus dataset evaluation completed!
echo ===============================================================================
echo.

REM =============================================================================
REM MEDBULLETS DATASET - Leadership Configuration
REM =============================================================================
echo [MedBullets] Running Leadership configuration...
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/leadership --leadership --seed 111 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/leadership --leadership --seed 222 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/leadership --leadership --seed 333 --n-max 2

REM =============================================================================
REM MEDBULLETS DATASET - Closed-loop Configuration
REM =============================================================================
echo [MedBullets] Running Closed-loop configuration...
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/closedloop --closedloop --seed 111 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/closedloop --closedloop --seed 222 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/closedloop --closedloop --seed 333 --n-max 2

REM =============================================================================
REM MEDBULLETS DATASET - Mutual Monitoring Configuration
REM =============================================================================
echo [MedBullets] Running Mutual Monitoring configuration...
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/mutual_monitoring --mutual --seed 111 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/mutual_monitoring --mutual --seed 222 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/mutual_monitoring --mutual --seed 333 --n-max 2

REM =============================================================================
REM MEDBULLETS DATASET - Shared Mental Model Configuration
REM =============================================================================
echo [MedBullets] Running Shared Mental Model configuration...
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/shared_mental_model --mental --seed 111 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/shared_mental_model --mental --seed 222 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/shared_mental_model --mental --seed 333 --n-max 2

REM =============================================================================
REM MEDBULLETS DATASET - Team Orientation Configuration
REM =============================================================================
echo [MedBullets] Running Team Orientation configuration...
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/team_orientation --orientation --seed 111 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/team_orientation --orientation --seed 222 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/team_orientation --orientation --seed 333 --n-max 2

REM =============================================================================
REM MEDBULLETS DATASET - Mutual Trust Configuration
REM =============================================================================
echo [MedBullets] Running Mutual Trust configuration...
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/mutual_trust --trust --seed 111 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/mutual_trust --trust --seed 222 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/mutual_trust --trust --seed 333 --n-max 2

REM =============================================================================
REM MEDBULLETS DATASET - All Features Configuration
REM =============================================================================
echo [MedBullets] Running All Features configuration...
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/all_features --leadership --closedloop --mutual --mental --orientation --trust --seed 111 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/all_features --leadership --closedloop --mutual --mental --orientation --trust --seed 222 --n-max 2
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/all_features --leadership --closedloop --mutual --mental --orientation --trust --seed 333 --n-max 2

echo.
echo ===============================================================================
echo MedBullets dataset evaluation completed!
echo ===============================================================================
echo.

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



echo "All special set experiments completed!"
pause

