
class Config:

    # Experiment logging specifics
    NAME = None
    DATE = None
    SEED = None
    WEIGHTS_PATH = "./"
    DEVICE = None

    LEARNING_RATE = 1e-3
    GAMMA = 0.99
    TOTAL_TIMESTEPS = 5e5
    EVAL_EPS = 10
    NORM_OBS = False
    UPDATE_FREQ = 100
    BATCH_SIZE = 20
    EPOCHS_PER_UPDATE = 4
    MAX_GRAD_NORM = 0.5
    MAX_TIMESTEPS_PER_EPISODE = 200
    USE_LSTM = True
    # Continuous spaces learn distribution
    LOG_STD = -0.5
    LEARNED_STD = True

    # Algorithm specific
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    CLIP_RANGE = 0.25
    USE_GAE = True
    GAE_LAMBDA = 0.95