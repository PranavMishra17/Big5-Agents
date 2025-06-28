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




REM =============================================================================
REM SYMCAT DATASET - Leadership Configuration
REM =============================================================================
echo [SymCat] Running Leadership configuration...
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/leadership --leadership --seed 111 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/leadership --leadership --seed 222 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/leadership --leadership --seed 333 --n-max 3

REM =============================================================================
REM SYMCAT DATASET - Closed-loop Configuration
REM =============================================================================
echo [SymCat] Running Closed-loop configuration...
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/closedloop --closedloop --seed 111 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/closedloop --closedloop --seed 222 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/closedloop --closedloop --seed 333 --n-max 3

REM =============================================================================
REM SYMCAT DATASET - Mutual Monitoring Configuration
REM =============================================================================
echo [SymCat] Running Mutual Monitoring configuration..
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/mutual_monitoring --mutual --seed 111 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/mutual_monitoring --mutual --seed 222 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/mutual_monitoring --mutual --seed 333 --n-max 3

REM =============================================================================
REM SYMCAT DATASET - Shared Mental Model Configuration
REM =============================================================================
echo [SymCat] Running Shared Mental Model configuration...
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/shared_mental_model --mental --seed 111 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/shared_mental_model --mental --seed 222 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/shared_mental_model --mental --seed 333 --n-max 3

REM =============================================================================
REM SYMCAT DATASET - Team Orientation Configuration
REM =============================================================================
echo [SymCat] Running Team Orientation configuration...
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/team_orientation --orientation --seed 111 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/team_orientation --orientation --seed 222 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/team_orientation --orientation --seed 333 --n-max 3

REM =============================================================================
REM SYMCAT DATASET - Mutual Trust Configuration
REM =============================================================================
echo [SymCat] Running Mutual Trust configuration...
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/mutual_trust --trust --seed 111 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/mutual_trust --trust --seed 222 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/mutual_trust --trust --seed 333 --n-max 3

REM =============================================================================
REM SYMCAT DATASET - All Features Configuration
REM =============================================================================
echo [SymCat] Running All Features configuration...
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/all_features --leadership --closedloop --mutual --mental --orientation --trust --seed 111 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/all_features --leadership --closedloop --mutual --mental --orientation --trust --seed 222 --n-max 3
python dataset_runner.py --dataset symcat --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/symcat3/all_features --leadership --closedloop --mutual --mental --orientation --trust --seed 333 --n-max 3

echo.
echo ===============================================================================
echo SymCat dataset evaluation completed!
echo ===============================================================================
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
::python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/leadership --leadership --seed 111 --n-max 2
::python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/leadership --leadership --seed 222 --n-max 2
::python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/leadership --leadership --seed 333 --n-max 2

REM =============================================================================
REM MEDBULLETS DATASET - Closed-loop Configuration
REM =============================================================================
echo [MedBullets] Running Closed-loop configuration...
::python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/closedloop --closedloop --seed 111 --n-max 2
::python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/closedloop --closedloop --seed 222 --n-max 2
::python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --output-dir ./results/medbullets2/closedloop --closedloop --seed 333 --n-max 2

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
