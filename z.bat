@echo off
REM DDXPlus and MedBullets Dataset Commands - Best Components by Team Size

echo Running DDXPlus Dataset Commands...
echo.

REM DDXPlus Dataset Commands
REM n = 2: Mutual Trust, Shared Mental Model (assuming "mutual" = Mutual Trust, "mental" = Shared Mental Model)

echo DDXPlus n=2 commands (DDX_sp2 folder)
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --n-max 2 --seed 111 --output-dir ./results/DDX_sp2 --trust --mental
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --n-max 2 --seed 222 --output-dir ./results/DDX_sp2 --trust --mental
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --n-max 2 --seed 333 --output-dir ./results/DDX_sp2 --trust --mental

echo.
echo DDXPlus n=3 commands (DDX_sp3 folder)
REM n = 3: Closed-loop, Leadership, Shared Mental Model
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --n-max 3 --seed 111 --output-dir ./results/DDX_sp3 --closedloop --leadership --mental
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --n-max 3 --seed 222 --output-dir ./results/DDX_sp3 --closedloop --leadership --mental
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --n-max 3 --seed 333 --output-dir ./results/DDX_sp3 --closedloop --leadership --mental

echo.
echo DDXPlus n=4 commands (DDX_sp4 folder)
REM n = 4: Mutual Trust, Shared Mental Model
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --n-max 4 --seed 111 --output-dir ./results/DDX_sp4 --trust --mental
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --n-max 4 --seed 222 --output-dir ./results/DDX_sp4 --trust --mental
python dataset_runner.py --dataset ddxplus --num-questions 50 --recruitment --recruitment-method intermediate --n-max 4 --seed 333 --output-dir ./results/DDX_sp4 --trust --mental

echo.
echo.
echo Running MedBullets Dataset Commands...
echo.

REM MedBullets Dataset Commands
REM n = 2: Mutual Trust, Team Orientation (assuming "mutual" = Mutual Trust, "orientation" = Team Orientation)

echo MedBullets n=2 commands (MED_sp2 folder)
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --n-max 2 --seed 111 --output-dir ./results/MED_sp2 --trust --orientation
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --n-max 2 --seed 222 --output-dir ./results/MED_sp2 --trust --orientation
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --n-max 2 --seed 333 --output-dir ./results/MED_sp2 --trust --orientation

echo.
echo MedBullets n=3 commands (MED_sp3 folder)
REM n = 3: Mutual Trust, Team Orientation, Closed-loop (assuming "nutual" is typo for "mutual")
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --n-max 3 --seed 111 --output-dir ./results/MED_sp3 --trust --orientation --closedloop
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --n-max 3 --seed 222 --output-dir ./results/MED_sp3 --trust --orientation --closedloop
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --n-max 3 --seed 333 --output-dir ./results/MED_sp3 --trust --orientation --closedloop

echo.
echo MedBullets n=4 commands (MED_sp4 folder)
REM n = 4: Shared Mental Model, Closed-loop, Team Orientation
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --n-max 4 --seed 111 --output-dir ./results/MED_sp4 --mental --closedloop --orientation
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --n-max 4 --seed 222 --output-dir ./results/MED_sp4 --mental --closedloop --orientation
python dataset_runner.py --dataset medbullets --num-questions 50 --recruitment --recruitment-method intermediate --n-max 4 --seed 333 --output-dir ./results/MED_sp4 --mental --closedloop --orientation

echo.
echo All commands completed!
pause