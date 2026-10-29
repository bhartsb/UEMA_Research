# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Add self perception without changing any existing learned tensor or Adam state."""
import argparse
from .prepare_continuity import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint');p.add_argument('--out',required=True)
    p.add_argument('--episode-storage',help='Keep the physical companion on the monitored recording drive')
    a=p.parse_args();prepare(a.checkpoint,a.out,extension='self',episode_storage=a.episode_storage)
