#! /bin/bash

python3 -m venv chess_game &&
source chess_game/bin/activate &&

pip install python-chess &&
pip install pygame-ce==2.5.4 &&
pip install torch --index-url https://download.pytorch.org/whl/cpu &&

python game.py