# YOLO26n-BDD100K

This folder is the model-specific entry point for the Task 2 YOLO26n run.
The shared implementation remains at `../task2_vru_tracking.py`; keeping the
launcher and artifacts here prevents YOLO26n results from mixing with YOLO26m.

Use `run_yolo26n.ps1` to write Task 2 outputs under `task2_yolo26n/results/`.
The checkpoint is kept outside Git at `../session/yolo26n-bdd100k/`.
