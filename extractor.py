import sys, getopt
from analyzer.analyzer import Analyzer
from model.models import MainVocalModel, VADModel

BIN_COEF = 0
SOUND_SMOOTHING = 0
VOCAL_SENSITIVITY = 0
FILE_NAME = ''
FILE_OUT = ''


def handler(argv):
    global BIN_COEF, SOUND_SMOOTHING, VOCAL_SENSITIVITY, FILE_NAME, FILE_OUT
    try:
        list = getopt.gnu_getopt(argv, "filenameb:s:v:o:")
        print(list)
        for entry in list[0]:
            if entry[0] == '-b':
                BIN_COEF = float(entry[1])
            elif entry[0] == '-v':
                VOCAL_SENSITIVITY = float(entry[1])
            elif entry[0] == '-s':
                SOUND_SMOOTHING = float(entry[1])
            elif entry[0] == '-o':
                FILE_OUT = entry[1]
        FILE_NAME = list[1][1]
    except:
        print('Usage: <filename> -b <binary_coef> -v <vocal_sens> -s <sound_smooth> -o <output_path>')
        exit(1)


handler(sys.argv)

model = MainVocalModel(weights="weights/vocal_model_weights.hdf5")
vad_model = VADModel(weights="weights/vad_model_weights.hdf5")
analyzer = Analyzer(model=model, vad_model=vad_model)

analyzer.extract_vocals(
    FILE_NAME,
    FILE_OUT,
    binarization_coeff=BIN_COEF,
    smooth_coeff=SOUND_SMOOTHING,
    vocal_sensitivity=VOCAL_SENSITIVITY
)
