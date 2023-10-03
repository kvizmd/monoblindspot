import os

import matplotlib

# plt.show is available by running `GUI=1 python ...`.
if os.environ.get('GUI', False):
    print('Enable Matplotlib GUI')
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')

from .config import load_config
from .datasets import \
    build_bs_dataset, \
    build_bsgen_dataset, \
    build_depth_dataset, \
    build_exporter
from .losses import build_criterion
from .models import build_model
from .ogm_models import build_integrator
from .utils import fix_random_state, get_random_seed

from .runners import \
    DepthTrainer, \
    DepthEvaluator, \
    BlindSpotGenerator, \
    BlindSpot2DTrainer, \
    BlindSpotEvaluator, \
    create_parallel_run
