class CFG:

    SEED = 5252
    LR = 1e-3
    BETAS = (0.9, 0.999)
    DECAY = 0.0001
    BATCH_SIZE = 2048
    MAX_EPOCHS = 64
    WINDOW_SIZE = 60
    WINDOW_GIVEN_LIST = [39, 44, 49, 54, 59, 64, 69, 74, 79]
<<<<<<< HEAD
    WINDOW_SIZE_LIST = [40, 45, 50, 55, 60, 65, 70, 75, 80]
    SEQUENTIAL_VALID_COLUMNS_IN_TRAIN_DATASET = ["C01", "C03", "C04", "C05", "C06", "C07", "C08", "C11",  "C12", "C13", "C14", "C15", "C16", "C17", "C20", "C21", "C23", "C24", "C25", "C27", "C28", "C30", "C31", "C32", "C33", "C34", "C35", "C37", "C40", "C41", "C42", "C43", "C44", "C45", "C46", "C47", "C48", "C50", "C51", "C53", "C54",  "C56", "C57", "C58", "C59", "C60", "C61",  "C62", "C64",  "C65", "C66", "C67", "C68", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C77", "C78", "C79", "C80", "C81", "C83", "C84", "C86"]
    CATEGORICAL_VALID_COLUMNS_IN_TRAIN_DATASET = ["C02", "C09", "C10", "C18", "C19", "C22", "C26", "C29", "C36", "C38", "C39", "C49", "C52", "C55", "C63", "C69", "C82", "C85"]
=======
    WINDOW_SIZE_LIST = [40, 46, 50, 54, 60, 66, 70, 74, 80]
>>>>>>> origin/0831

    # GRU & LSTM
    NUM_LAYERS = 3
    HIDDEN_SIZE = 100
    BIDIRECTIONAL = True
    DROPOUT = 0.1
    THRESHOLD = 0.04
    THRESHOLD_RANGE = [
        0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035,
        0.036, 0.037, 0.038, 0.039, 0.04
    ]
    # STEPS = 1000

    # PYTORCH
    SWA = True
