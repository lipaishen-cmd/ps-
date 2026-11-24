"""PCB external-method adaptations; see README for fidelity boundaries."""
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", message=".*torch\\.jit\\.script.*deprecated.*", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/'environment', ROOT/'agent'):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
METHODS = {'am_a2c':'AM-A2C-PCB', 'transformer_a3c':'TRANSFORMER-A3C-PCB',
           'etdqn':'ETDQN-PCB', 'iddqn':'ATTENTION-DDQN-PCB'}
