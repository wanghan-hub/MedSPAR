"""Standalone Stage0 evaluation, sharing the native-grid scoring protocol."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.eval_stage1_tc_mve import main
if __name__ == '__main__': main(stage0=True)
