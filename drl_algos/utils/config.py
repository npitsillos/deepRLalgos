
class Config:

    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    SEED = None
    NAME = None
    TOTAL_TIMESTEPS = 1e6
    EVAL_EPS = 10
    DEVICE = None
    NORM_OBS = True
    UPDATE_FREQ = 20
    BATCH_SIZE = 5
    EPOCHS_PER_UPDATE = 4
    MAX_GRAD_NORM = 0.5
    WEIGHTS_PATH = "./"
    MAX_TIMESTEPS_PER_EPISODE = 300
    SEED = 2

    # Continuous spaces learn distribution
    LOG_STD = -0.5
    LEARNED_STD = True

    # Algorithm specific
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    CLIP_RANGE = 0.2
    USE_GAE = False
    GAE_LAMBDA = 0.95