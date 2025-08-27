@echo off
title Big5-Agents System Check
color 0E
echo.
echo ================================================================
echo               BIG5-AGENTS SYSTEM CHECK
echo            Pre-Demo Configuration Validation
echo ================================================================
echo.
echo This script validates your system setup before running demos.
echo.
pause

echo Checking Python installation...
python --version
echo.

echo Checking required Python packages...
python -c "import openai, datasets, torch; print('✓ Core packages installed')" 2>nul || echo "❌ Missing packages - run: pip install -r requirements.txt"
echo.

echo Checking configuration file...
python -c "import config; print(f'✓ Config loaded: {len(config.get_all_deployments())} deployment(s) configured')" 2>nul || echo "❌ Configuration error"
echo.

echo Checking dataset availability...
python -c "from datasets import load_dataset; print('✓ Datasets library ready')" 2>nul || echo "❌ Dataset loading issues"
echo.

echo Current directory structure:
dir /B *.py | findstr /R "main dataset_runner config simulator" >nul && echo "✓ Core Python files present" || echo "❌ Missing core files"
echo.

echo Checking log and output directories...
if exist "logs" (echo ✓ Logs directory ready) else (echo ❌ Logs directory missing)
if exist "output" (echo ✓ Output directory ready) else (echo ❌ Output directory missing)
echo.

echo Testing basic system functionality...
echo Running: python config.py
python config.py
echo.

echo ================================================================
echo                    SYSTEM CHECK COMPLETE
echo ================================================================
echo.
echo If all checks show ✓, your system is ready for demos.
echo If any checks show ❌, please fix those issues first.
echo.
echo Ready to run demos:
echo - quick_demo.bat      (2-3 minutes)
echo - demo_showcase.bat   (15-20 minutes full demo)
echo.
echo ================================================================
pause