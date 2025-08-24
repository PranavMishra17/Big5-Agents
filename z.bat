@echo off
REM =============================================================================
REM FULL-SCALE DATASET EVALUATION WITH DYNAMIC SELECTION
REM All datasets with --enable-dynamic-selection flag
REM Single seed for full evaluation (1000 questions each)
REM =============================================================================


echo Starting full-scale evaluation with dynamic selection...
echo Datasets: All available medical datasets
echo Mode: Dynamic selection enabled
echo Seed: 42 (for reproducibility)
echo Questions: 1000 per dataset (where available)
echo.


REM MedBullets Dataset - Dynamic Selection (308 questions available)
echo.
echo Running MedBullets dataset with dynamic selection (308 questions)...
echo Available: 308 questions
python dataset_runner.py --dataset medbullets --num-questions 308 --enable-dynamic-selection --seed 42 --output-dir ./results/dynamic/medbullets

REM MMLU-Pro Medical Dataset - Dynamic Selection (823 questions available)
echo.
echo Running MMLU-Pro Medical dataset with dynamic selection (823 questions)...
echo Available: 823 questions
python dataset_runner.py --dataset mmlupro-med --num-questions 823 --enable-dynamic-selection --seed 42 --output-dir ./results/dynamic/mmlupro-med


REM PubMedQA Dataset - Dynamic Selection (1,000 questions available)
echo.
echo Running PubMedQA dataset with dynamic selection (1000 questions)...
echo Available: 1,000 questions
python dataset_runner.py --dataset pubmedqa --num-questions 1000 --enable-dynamic-selection --seed 42 --output-dir ./results/dynamic/pubmedqa

