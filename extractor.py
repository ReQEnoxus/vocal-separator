from analyzer.analyzer import Analyzer
from model.models import Model

model = Model(weights="weights/vocal_model_weights.hdf5")
analyzer = Analyzer(model=model)

analyzer.extract_vocals("test/input/1.wav", "test/output/1.wav", binarization_coeff=0.19)