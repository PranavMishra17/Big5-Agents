@echo off
REM =============================================================================
REM SIMPLE SLM EVALUATION - 50 QUESTIONS PER DATASET
REM All datasets with 50 questions each for quick testing
REM Results stored in SLM_Results/ directory
REM =============================================================================

echo Starting simple SLM evaluation with 50 questions per dataset...
echo Datasets: All available medical datasets
echo Mode: Simple 50 questions each
echo Seed: 42 (for reproducibility)
echo Questions: 50 per dataset
echo Results: SLM_Results/ directory
echo.

REM =============================================================================
REM MEDICAL TEXT-BASED DATASETS
REM =============================================================================

echo.
echo ===============================================================================
echo RUNNING MEDICAL TEXT-BASED DATASETS
echo ===============================================================================

REM MedMCQA Dataset - 50 questions
echo.
echo Running MedMCQA dataset (50 questions)...
echo Available: 182,822 questions
python dataset_runner.py --dataset medmcqa --num-questions 50 --seed 42 --output-dir ./SLM_Results/gemma/medmcqa

REM MedQA Dataset - 50 questions
echo.
echo Running MedQA dataset (50 questions)...
echo Available: 10,178 questions
python dataset_runner.py --dataset medqa --num-questions 50 --seed 42 --output-dir ./SLM_Results/gemma/medqa

REM PubMedQA Dataset - 50 questions
echo.
echo Running PubMedQA dataset (50 questions)...
echo Available: 1,000 questions
python dataset_runner.py --dataset pubmedqa --num-questions 50 --seed 42 --output-dir ./SLM_Results/gemma/pubmedqa

REM MMLU-Pro Medical Dataset - 50 questions
echo.
echo Running MMLU-Pro Medical dataset (50 questions)...
echo Available: 823 questions
python dataset_runner.py --dataset mmlupro-med --num-questions 50 --seed 42 --output-dir ./SLM_Results/gemma/mmlupro-med

REM MedBullets Dataset - 50 questions
echo.
echo Running MedBullets dataset (50 questions)...
echo Available: 308 questions
python dataset_runner.py --dataset medbullets --num-questions 50 --seed 42 --output-dir ./SLM_Results/gemma/medbullets

REM =============================================================================
REM VISION-BASED DATASETS
REM =============================================================================

echo.
echo ===============================================================================
echo RUNNING VISION-BASED DATASETS
echo ===============================================================================

REM Path-VQA Dataset - 50 questions
echo.
echo Running Path-VQA dataset (50 questions)...
python dataset_runner.py --dataset path_vqa --num-questions 50 --seed 42 --output-dir ./SLM_Results/gemma/path_vqa

REM PMC-VQA Dataset - 50 questions
echo.
echo Running PMC-VQA dataset (50 questions)...
python dataset_runner.py --dataset pmc_vqa --num-questions 50 --seed 42 --output-dir ./SLM_Results/gemma/pmc_vqa

REM =============================================================================
REM CONDITIONAL DATASETS
REM =============================================================================

echo.
echo ===============================================================================
echo RUNNING CONDITIONAL DATASETS (if available)
echo ===============================================================================

REM DDXPlus Dataset - 50 questions
echo.
echo Running DDXPlus dataset (50 questions)...
echo Available: 1,025,602 questions
python dataset_runner.py --dataset ddxplus --num-questions 50 --seed 42 --output-dir ./SLM_Results/gemma/ddxplus

REM =============================================================================
REM EVALUATION SUMMARY
REM =============================================================================
echo.
echo ===============================================================================
echo SIMPLE SLM EVALUATION COMPLETED!
echo ===============================================================================
echo.
echo Summary:
echo - Datasets evaluated: 8 (MedMCQA, MedQA, PubMedQA, MMLU-Pro Medical, MedBullets, Path-VQA, PMC-VQA, DDXPlus)
echo - Mode: Simple 50 questions each
echo - Seed used: 42 (for reproducibility)
echo - Questions per dataset: 50
echo - Total questions evaluated: 400
echo.
echo Dataset Sizes Available:
echo - MedMCQA: 182,822 questions (using 50)
echo - MedQA: 10,178 questions (using 50)
echo - PubMedQA: 1,000 questions (using 50)
echo - MMLU-Pro Medical: 823 questions (using 50)
echo - MedBullets: 308 questions (using 50)
echo - Path-VQA: ~32,000 questions (using 50)
echo - PMC-VQA: ~32,000 questions (using 50)
echo - DDXPlus: 1,025,602 questions (using 50)
echo.
echo Results saved in:
echo   ./SLM_Results/[dataset_name]/
echo.
echo Simple Setup Features:
echo - Fixed 50 questions per dataset for consistent testing
echo - No dynamic selection complexity
echo - Standard team formation
echo - Quick evaluation for SLM model testing
echo.
echo Next steps:
echo 1. Check results in ./SLM_Results/ directories
echo 2. Analyze SLM model performance across datasets
echo 3. Compare Vertex AI vs OpenAI performance
echo 4. Review token usage and cost efficiency
echo 5. Validate SLM integration functionality
echo.
echo Evaluation complete! Check the SLM_Results directories for detailed analysis.

pause
