#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:36:19 2026

@author: lukethomas
"""
import pprint

from pathlib import Path
from process_data import ProcessData  # this must match your main file name in code/

ROOT = Path(__file__).resolve().parents[1]      # FORMAL/
DATA = ROOT / "data" / "example_data_1.dat"     # <-- change to your real filename

results = (ProcessData(str(DATA)))
pprint.pprint(results)