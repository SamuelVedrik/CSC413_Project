from models.crnn import CRNN
from models.convnet import ConvNet
from models.mccrnn import MCCRNN

# ==== CONVNET ====
CONV_MODEL_OPTS = dict(
    layer_opts=[
        dict(in_channels=1, out_channels=64),
        dict(in_channels=64, out_channels=64),
        dict(in_channels=64, out_channels=128),
        dict(in_channels=128, out_channels=256),
        ],
    output_size=10
)

# ==== CRNN ====
CRNN_MODEL_OPTS = dict(
    layer_opts=[
        dict(in_channels=1, out_channels=32),
        dict(in_channels=32, out_channels=64),
        dict(in_channels=64, out_channels=64, maxpool_kernel=(4, 2)),
        dict(in_channels=64, out_channels=128),
        dict(in_channels=128, out_channels=256),
        ],
    gru_hidden_size=30,
    output_size=10
)

# ==== MCCRNN =====
MCCRNN_MODEL_OPTS = dict(
    column1_opts=[
        dict(in_channels=1, out_channels=48, kernel_size=(3, 13), padding=(1, 6), stride=1, maxpool_kernel=(4, 4)),
        dict(in_channels=48, out_channels=64, kernel_size=(3, 7), padding=(1, 3), stride=1, maxpool_kernel=(4, 2)),
        ],
    column2_opts=[
        dict(in_channels=1, out_channels=48, kernel_size=(13, 3), padding=(6, 1), stride=1, maxpool_kernel=(4, 4)),
        dict(in_channels=48, out_channels=64, kernel_size=(7, 3), padding=(3, 1), stride=1, maxpool_kernel=(4, 2)),
    ],
    combined_opts = [
        dict(in_channels=128, out_channels=256, maxpool_kernel=(2, 4))
    ],
    gru_hidden_size=30,
    output_size=10
)

OPTIONS = {
    "convnet": dict(
        model_class = ConvNet,
        model_opts = CONV_MODEL_OPTS
    ),
    "crnn": dict(
        model_class = CRNN,
        model_opts = CRNN_MODEL_OPTS
    ),
    "mccrnn": dict(
        model_class = MCCRNN,
        model_opts = MCCRNN_MODEL_OPTS
    )
}

