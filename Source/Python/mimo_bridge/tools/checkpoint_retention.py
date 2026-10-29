# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Plan or maintain registered production snapshot storage without stopping learning."""
import argparse
import json
import time
from pathlib import Path
from lem.training.retention import maintain


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--apply',action='store_true',help='Retire only unprotected, enrolled checkpoints in the reviewed policy')
    parser.add_argument('--watch',action='store_true',help='Recheck while the current trainer continues; does not restart training')
    parser.add_argument('--interval',type=float,default=60.)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    if args.interval<15:parser.error('retention interval must be at least15 seconds')
    while True:
        try:report=maintain(args.policy,apply=args.apply)
        except Exception as error:
            if not args.watch:raise
            report=dict(applied=False,error=str(error),timestamp=time.time(),retired=[])
        if args.report:
            temporary=args.report.with_suffix('.tmp')
            temporary.write_text(json.dumps(report,indent=2),encoding='utf-8');temporary.replace(args.report)
        print(json.dumps(report),flush=True)
        if not args.watch:return 0
        time.sleep(args.interval)


if __name__=='__main__':raise SystemExit(main())
