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

model = MainVocalModel(
    weights="E:\\Projects\\vocal-separator\\2k22\\main_artifacts\\7d550238-40ae-485a-93b1-00f17902aded\\weights-0.66.hdf5"
)
vad_model = VADModel(
    weights="E:\\Projects\\vocal-separator\\2k22\\artifacts\\b66d5b6e-7bb0-4b7a-977d-de503ddf3452\\\weights-0.85.hdf5",
    model_path="E:\\Projects\\vocal-separator\\2k22\\artifacts\\b66d5b6e-7bb0-4b7a-977d-de503ddf3452\\vad-model-b66d5b6e-7bb0-4b7a-977d-de503ddf3452.h5"
)
analyzer = Analyzer(model=model, vad_model=vad_model)

analyzer.extract_vocals(
    FILE_NAME,
    FILE_OUT,
    binarization_coeff=BIN_COEF,
    smooth_coeff=SOUND_SMOOTHING,
    vocal_sensitivity=VOCAL_SENSITIVITY
)
