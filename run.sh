#!/bin/bash
python3 /app/interpolate_poses.py /Data/trajectory.txt /Data/rgb

python3 /app/data_organizer.py --path /Data

python3 /app/reconstruction_system/run_system.py --make --register --refine --integrate --config /Data/ivero.json