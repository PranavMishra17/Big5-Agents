@echo off
title Big5-Agents Multi-Agent System Demo
color 0A
echo.
echo ================================================================
echo           BIG5-AGENTS MULTI-AGENT SYSTEM DEMO
echo        TeamMedAgents: Medical Decision-Making Framework
echo ================================================================
echo.
echo This demo showcases our multi-agent collaboration system based on
echo the Big Five teamwork model for medical decision-making tasks.
echo.
echo Key Features:
echo - Dynamic agent recruitment with medical specialists
echo - Modular teamwork components (Leadership, Communication, etc.)
echo - Multiple decision aggregation methods
echo - Support for text and vision-based medical datasets
echo.
pause


REM =============================================================================
REM Demo: Medical MCQ with Basic Teamwork Components
REM =============================================================================
cls
echo.
echo ================================================================
echo                 DEMO 2: MEDICAL MULTIPLE CHOICE
echo            MedMCQA Dataset with Core Teamwork Features
echo ================================================================
echo.
echo Medical entrance exam questions with teamwork components:
echo - Dataset: MedMCQA (Indian medical entrance exams)
echo - Features: Closed-loop Communication + Mutual Monitoring
echo - Team: Medical specialist recruitment (intermediate)
echo - Questions: 5 medical MCQ questions
echo.
echo Command: python dataset_runner.py --dataset medmcqa --num-questions 5 --closedloop --mutual --recruitment --recruitment-method intermediate --recruitment-pool medical --output-dir demo/medmcqa_teamwork
echo.
pause
python dataset_runner.py --dataset medmcqa --num-questions 5 --closedloop --mutual --recruitment --recruitment-method intermediate --recruitment-pool medical --output-dir demo/medmcqa_teamwork
echo.
echo ================================================================
echo Demo 2 Complete! Observed specialist recruitment and
echo collaborative medical reasoning with feedback loops.
echo ================================================================
pause

REM =============================================================================
REM Demo: Vision-Enabled Medical Questions - Advanced Recruitment
REM =============================================================================
cls
echo.
echo ================================================================
echo               DEMO 3: MEDICAL VISION ANALYSIS
echo         PMC-VQA Dataset with Advanced Agent Recruitment
echo ================================================================
echo.
echo Medical image analysis with advanced multi-agent collaboration:
echo - Dataset: PMC-VQA (PubMed Central Visual Question Answering)
echo - Features: All Big5 teamwork components
echo - Team: Advanced recruitment with vision specialists
echo - Questions: 3 medical image questions
echo.
echo Command: python dataset_runner.py --dataset pmc_vqa --num-questions 3 --leadership --closedloop --mutual --mental --orientation --trust --recruitment --recruitment-method advanced --recruitment-pool medical --n-max 5 --output-dir demo/vision_advanced
echo.
pause
python dataset_runner.py --dataset pmc_vqa --num-questions 3 --leadership --closedloop --mutual --mental --orientation --trust --recruitment --recruitment-method advanced --recruitment-pool medical --n-max 5 --output-dir demo/vision_advanced
echo.
echo ================================================================
echo Demo 3 Complete! Advanced vision-enabled collaboration with
echo specialized medical image analysis and comprehensive teamwork.
echo ================================================================
pause


REM =============================================================================
REM Demo: Dynamic Selection vs Static Configuration
REM =============================================================================
cls
echo.
echo ================================================================
echo              DEMO 7: DYNAMIC vs STATIC COMPARISON
echo         AI-Driven Team Configuration vs Fixed Setup
echo ================================================================
echo.
echo Comparing adaptive AI selection against fixed configuration:
echo - Dataset: DDXPlus (Clinical diagnosis cases)
echo - Features: Dynamic selection vs static leadership
echo - Team: AI determines optimal team composition
echo - Questions: 5 diagnostic scenarios
echo.
echo First: Dynamic AI-driven configuration
echo Command: python dataset_runner.py --dataset ddxplus --num-questions 5 --enable-dynamic-selection --recruitment-pool medical --output-dir demo/dynamic_selection
echo.
pause
python dataset_runner.py --dataset ddxplus --num-questions 5 --enable-dynamic-selection --recruitment-pool medical --output-dir demo/dynamic_selection
echo.
echo Now: Static configuration with fixed teamwork components
echo Command: python dataset_runner.py --dataset ddxplus --num-questions 5 --disable-dynamic-selection --leadership --closedloop --recruitment --recruitment-method intermediate --recruitment-pool medical --output-dir demo/static_config
echo.
pause
python dataset_runner.py --dataset ddxplus --num-questions 5 --disable-dynamic-selection --leadership --closedloop --recruitment --recruitment-method intermediate --recruitment-pool medical --output-dir demo/static_config
echo.
echo ================================================================
echo Demo 7 Complete! Compared dynamic AI selection with static
echo configuration, showing adaptation benefits.
echo ================================================================
pause

REM =============================================================================
REM Demo Summary
REM =============================================================================
cls
echo.
echo ================================================================
echo                    DEMO SHOWCASE COMPLETE
echo              Big5-Agents Multi-Agent System
echo ================================================================
echo.
echo You have experienced:
echo.
echo ✓ Medical multiple choice questions (MedMCQA)
echo ✓ Medical image analysis (PMC-VQA, Path-VQA)
echo ✓ Clinical research questions (PubMedQA)
echo ✓ Trust-based collaboration mechanisms
echo ✓ Comprehensive configuration comparison
echo ✓ Dynamic vs static team selection
echo ✓ Specialized pathology image analysis
echo.
echo Key Capabilities Demonstrated:
echo - Big Five teamwork model implementation
echo - Dynamic agent recruitment and specialization
echo - Multi-modal processing (text + vision)
echo - Multiple decision aggregation methods
echo - Comprehensive medical domain coverage
echo - Scalable parallel processing architecture
echo.
echo Results Location:
echo - Demo results: ./demo/ directory (organized by scenario)
echo - Detailed logs: ./logs/ directory  
echo - Performance metrics: JSON files in each demo subfolder
echo.
echo ================================================================
echo Thank you for exploring our Big5-Agents system!
echo For questions: Check README.md or research paper
echo ================================================================
echo.
pause