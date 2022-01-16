# vocal-separator
CNN for separating vocal track from a composed song

## Usage

`extractor.py <filename> -b <bin_coefficient> -v <vocal_sensitivity> -s <vocal_smooth> -o <output_path>`

* `<filename>` - path to input file
* `<bin_coefficient>` - binarization coefficient - a value, starting from which the probability is considered to be high enough to put 1 into resulting vector
* `<vocal_sensitivity>` - binarization coefficient analog for vocal activity detector - the level of certainty from which the vocal is considered to be present in current frame
* `<vocal_smooth>` - integer number of subsequent frames which are considered to have vocals after the frame where vocals were detected
* `<output_path>` - path to the output file

## Tests

There are [several test samples](https://github.com/ReQEnoxus/vocal-separator/tree/master/tests) to evaluate the model. Parameters used to generate outputs:
1. `extractor.py ./tests/input/before1.wav -b 0.285 -v 0.5 -s 25 -o ./tests/output/after1.wav`
2. `extractor.py ./tests/input/before2.wav -b 0.218 -v 0.89 -s 20 -o ./tests/output/after2.wav`
3. `extractor.py ./tests/input/before3.wav -b 0.212 -v 0.89 -s 20 -o ./tests/output/after3.wav`
