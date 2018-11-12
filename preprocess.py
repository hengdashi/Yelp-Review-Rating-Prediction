import numpy as np
import pandas as pd
from constants import *
from pathlib import Path

from preprocessor import Preprocessor

if __name__ == "__main__":
    datafolder = Path(data_path)
    preprocessor = Preprocessor(datafolder)
    preprocessor.preprocess_bus()
    preprocessor.preprocess_users()
    preprocessor.preprocess_reviews()
