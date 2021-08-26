class CFG:

    SEED = 5252
    LR = 1e-3
    BETAS = (0.9, 0.999)
    DECAY = 0.0001
    BATCH_SIZE = 1024
    MAX_EPOCHS = 64
    WINDOW_SIZE = 60
    WINDOW_GIVEN_LIST = [39, 44, 49, 54, 59, 64, 69, 74, 79]
    WINDOW_SIZE_LIST = [40, 45, 50, 55, 60, 65, 70, 75, 80]

    # GRU & LSTM
    NUM_LAYERS = 3
    HIDDEN_SIZE = 100
    BIDIRECTIONAL = True
    DROPOUT = 0
    # STEPS = 1000

    # PYTORCH
    SWA = True
