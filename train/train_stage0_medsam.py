"""Stage0: plain MedSAM, shared protocol and shared-parameter training policy."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.train_stage1 import main
if __name__ == '__main__': main(stage0=True)
