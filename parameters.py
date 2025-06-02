class EnvParams:
    SPECIES_AGENTS_RANGE = (3, 3)
    SPECIES_RANGE = (3, 5)
    TASKS_RANGE = (15, 50)
    MAX_TIME = 200
    TRAIT_DIM = 5
    DECISION_DIM = 30
    RENDER_MODE = "human"

class TrainParams:
    FORCE_MAX_OPEN_TASK = False
    POMO_SIZE = 10
    USE_GPU = False
    USE_GPU_GLOBAL = True
    AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
    TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM
    EMBEDDING_DIM = 128
    LR = 1e-5
    DECAY_STEP = 2e3
    RESET_OPT = False
    NUM_META_AGENT = 16
    NUM_GPU = 1
    EVALUATION_SAMPLES = 256
    BATCH_SIZE = 2048
    SUMMARY_WINDOW = 8
    INCREASE_DIFFICULTY = 20000
    EVALUATE = True

class SaverParams:
    FOLDER_NAME = 'save_5'
    TRAIN_PATH = f'train/{FOLDER_NAME}'
    SAVE = True
    MODEL_PATH = f'model/{FOLDER_NAME}'
    LOAD_MODEL = False
    LOAD_FROM = 'current'