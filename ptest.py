#!/usr/bin/env python3
# minimal_visualize_model.py
import os
import argparse
import time
import opensim as osim
from pathlib import Path

def main():
    # Load model and show
    model = osim.Model('models/OpenSim/LaiUhlrich2022.osim')

    state = model.initSystem()


if __name__ == "__main__":
    main()
