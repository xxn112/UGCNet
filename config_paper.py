import os
import torch

# This config is paper-consistent: it mirrors exactly the methodology
# described in the manuscript (Section 3). Differences from the original
# v22 implementation are intentional, so that code == paper:
#   * F = 4 single-resolution log curves (GR, AC, DEN, LLD), NOT 8ch dual-res
#   * 4 Transformer encoder layers, 2-layer GAT (4 heads then 1 head)
#   * L_ce = standard cross-entropy (no Focal modulation, no class weighting)
#   * No prediction post-processing (no mode filter, no monotonic correction)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, 'Step1_Simplified_Layers_Filtered')
COORD_FILE = os.path.join(PROJECT_ROOT, 'XYZGANCHAI.xlsx')
OUTPUT_DIR = os.path.join(THIS_DIR, 'results_paper')
MODEL_DIR = os.path.join(THIS_DIR, 'models')

# ------------------------------------------------------------------
# Data definition  (paper: F = 4, C = 4)
# ------------------------------------------------------------------
WELL_COLS = ['深度', 'AC', 'DEN', 'GR', 'LLD', '分层']
ALL_CURVE_COLS = ['GR', 'AC', 'DEN', 'LLD']   # paper order: GR, AC, DEN, LLD
N_FEATURES = 4                                # paper F = 4 (single resolution)
COORD_COLS = ['井名', 'X', 'Y', '补心海拔']
TARGET_ZONE = ['盐底', 'VI底']

LABEL_MAP = {'III': 0, 'IV': 1, 'V': 2, 'VI': 3}
INV_LABEL_MAP = {0: 'III', 1: 'IV', 2: 'V', 3: 'VI'}

FILTER_STRATEGY = 'keep_at_least_two'

# Graph construction thresholds (paper: d_ik < 5000 m, |z_i - z_k| < 100 m)
ALTITUDE_DIFF_THRESH = 100
PLANE_DIST_THRESH = 5000

MISSING_VALUES = [-9999.0, -999.0, -999.23, -999.25]

PHYSICAL_RANGES = {
    'AC': (100, 500),
    'DEN': (1.5, 3.5),
    'GR': (0, 500),
    'LLD': (0.1, 10000),
}

# ------------------------------------------------------------------
# Backbone hyper-parameters  (paper Implementation Settings)
# ------------------------------------------------------------------
HIDDEN_DIM = 128      # d = 128
NHEAD = 8             # 8 attention heads
NUM_LAYERS = 4        # 4 Transformer encoder layers
GNN_LAYERS = 2        # two-layer GAT (first 4 heads, second 1 head)

DEPTH_SCALE = 1000.0  # TVDSS positional-encoding scale

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
WELLS_PER_MB = 2
EPOCHS = 800
LR = 5e-4
K_FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

WARMUP_EPOCHS = 10
T_0_RESTART = 80
T_MULT_FACTOR = 2
LR_MIN = 1e-6

EARLY_STOPPING_PATIENCE = 80
VAL_FREQUENCY = 2
AUG_NOISE_STD = 0.01
WEIGHT_DECAY = 5e-5

# Paper: L_ce is standard cross-entropy -> no Focal, no class weighting.
FOCAL_GAMMA = 0.0
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

# ------------------------------------------------------------------
# Adaptive Sequence Constraint (ASC)
#   w = clamp( (1 - (H_asc)^t)^g , w_min, w_max ),  t=1.5, g=1.2
#   base_weight = lambda_asc = 10.0
# ------------------------------------------------------------------
ASC_CONFIG = {
    'base_weight': 10.0,     # lambda_asc
    'temperature': 1.5,      # t
    'min_confidence': 0.15,  # w_min
    'max_confidence': 1.0,   # w_max
    'exponent': 1.2,         # g
    'margin': 0.0,
}

# ------------------------------------------------------------------
# Adaptive Thickness Constraint (ATC)
#   w^atc = clamp( s^{t_atc}, w_min, w_max ),  t_atc = 1.0
#   delta_min = 3 sampling points (0.375 m at 0.125 m spacing)
#   base_weight = lambda_atc = 0.1
# ------------------------------------------------------------------
ATC_CONFIG = {
    'base_weight': 0.1,      # lambda_atc
    'temperature': 1.0,      # t_atc
    'min_confidence': 0.1,   # w_min
    'max_confidence': 1.0,   # w_max
    'exponent': 1.0,
    'min_layer_length': 3,   # delta_min
    'stability_window': 5,   # W
}
