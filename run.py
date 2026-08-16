#!/usr/bin/env python3
"""
Top-level entrypoint for Distillation Economics Suite.
Usage:
  python run.py --quick
  python run.py --teacher-rnd 150 --distill-queries 100000 --save-plots
"""

from distillation_economics.cli import main

if __name__ == "__main__":
    main()
