class CFG:

    SEED = 5252
    LR = 1e-3
    BETAS = (0.9, 0.999)
    DECAY = 0.0001
    BATCH_SIZE = 512
    MAX_EPOCHS = 512
    WINDOW_SIZE = 90

    # GRU & LSTM
    NUM_LAYERS = 3
    HIDDEN_SIZE = 100
    BIDIRECTIONAL = True
    DROPOUT = 0.0
    REVERSED = False
    # STEPS = 1000
