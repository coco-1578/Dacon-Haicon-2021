class CFG:

    SEED = 5252
    LR = 1e-3
    BETAS = (0.9, 0.999)
    DECAY = 0.0001
    BATCH_SIZE = 8192
    MAX_EPOCHS = 64
    WINDOW_SIZE = 60

    # GRU & LSTM
    NUM_LAYERS = 3
    HIDDEN_SIZE = 128
    BIDIRECTIONAL = True
    DROPOUT = 0.1
    # STEPS = 1000

    # PYTORCH
    SWA = True
