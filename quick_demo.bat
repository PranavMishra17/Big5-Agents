@echo off
title Quick Big5-Agents Demo
color 0B
echo.
echo ================================================================
echo                  QUICK BIG5-AGENTS DEMO
echo              (Fast 2-minute demonstration)
echo ================================================================
echo.
echo This is a condensed version for quick interviews/presentations.
echo Each demo uses minimal questions for faster execution.
echo.
pause

REM Quick Medical MCQ Demo
cls
echo.
echo ================================================================
echo          DEMO: MEDICAL AI COLLABORATION
echo        2 Medical Questions with Team Leadership
echo ================================================================
echo.
echo Demonstrating medical specialist collaboration:
echo - Medical entrance exam questions
echo - Team leadership coordination
echo - Specialist recruitment
echo.
echo Command: python dataset_runner.py --dataset medmcqa --num-questions 2 --leadership --recruitment --recruitment-method intermediate --recruitment-pool medical --output-dir demo/quick_medical
echo.
pause
python dataset_runner.py --dataset medmcqa --num-questions 2 --leadership --recruitment --recruitment-method intermediate --recruitment-pool medical --output-dir demo/quick_medical

REM Quick Vision Demo
cls
echo.
echo ================================================================
echo          DEMO: MEDICAL IMAGE ANALYSIS
echo        1 Vision Question with Advanced Teamwork
echo ================================================================
echo.
echo Demonstrating AI medical image analysis:
echo - Medical image question from research papers
echo - Vision-capable specialists
echo - Advanced collaboration features
echo.
echo Command: python dataset_runner.py --dataset pmc_vqa --num-questions 1 --leadership --closedloop --mutual --recruitment --recruitment-method advanced --recruitment-pool medical --output-dir demo/quick_vision
echo.
pause
python dataset_runner.py --dataset pmc_vqa --num-questions 1 --leadership --closedloop --mutual --recruitment --recruitment-method advanced --recruitment-pool medical --output-dir demo/quick_vision

REM Classic Ranking Task
cls
echo.
echo ================================================================
echo             DEMO: CLASSIC DECISION TASK
echo          NASA Lunar Survival Ranking
echo ================================================================
echo.
echo Demonstrating structured decision-making:
echo - Classic survival scenario
echo - Team coordination
echo - Multiple decision methods
echo.
echo Command: python main.py --leadership --recruitment --recruitment-method intermediate --output-dir demo/quick_ranking
echo.
pause
python main.py --leadership --recruitment --recruitment-method intermediate --output-dir demo/quick_ranking

cls
echo.
echo ================================================================
echo                  QUICK DEMO COMPLETE!
echo ================================================================
echo.
echo You've seen:
echo ✓ Medical specialist collaboration
echo ✓ Vision-enabled image analysis  
echo ✓ Classic decision-making tasks
echo ✓ Dynamic agent recruitment
echo ✓ Multiple decision aggregation methods
echo.
echo For full demonstration: run demo_showcase.bat
echo Results saved in: ./demo/ directory (organized by scenario)
echo.
echo ================================================================
pause