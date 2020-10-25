import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, dataset: str, verbose: int = 0,
                 test_split: float = 0.33):
        self._dataset = dataset
        self._verbose = verbose
        self._test_split = test_split

        self._datasets_folder = 'datasets'

    def load(self):
        if self._dataset == 'chinese_mnist':
            return self._load_cm()
        elif self._dataset == 'lego_figures':
            return self._load_lf()
        else:
            return None

    def _load_cm(self):
        DATASET = 'chinese_mnist'
        DATA_FILE = 'chinese_mnist.csv'

        DATA_FOLDER = 'data/data'
        FILE_FORMATTER = 'input_{}_{}_{}.jpg'

        DATA_FILE_PATH = os.path.join(self._datasets_folder, DATASET, DATA_FILE)
        DATA_FOLDER_PATH = os.path.join(self._datasets_folder, DATASET, DATA_FOLDER)

        df = pd.read_csv(DATA_FILE_PATH)

        data_size = df.shape[0]

        X = []
        y = []

        if self._verbose > 0:
            print("Loading data...")

        used_values = [1, 5, 8, 10]

        for i in range(data_size):
            if self._verbose > 0:
                print("Progress:", int(100 * i / data_size), "%", end='\r', flush=True)

            input_data = df.iloc[i]

            value = int(input_data['code'])
            if value not in used_values:
                continue

            input_image_path = os.path.join(DATA_FOLDER_PATH,
                                            FILE_FORMATTER
                                            .format(input_data['suite_id'],
                                                    input_data['sample_id'],
                                                    value))

            input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

            # scale down (???)
            # input_image = cv2.resize(input_image, (28, 28))

            input_image_vector = input_image.flatten()

            X.append(input_image_vector)
            y.append(value)

        X = np.array(X)
        y = np.array(y)

        X_norm = StandardScaler().fit_transform(X)

        if self._verbose > 0:
            print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X_norm, y,
                                                            test_size=self._test_split,
                                                            random_state=42)

        if self._verbose > 0:
            print("Done!")

        return X_train, X_test, y_train, y_test


    def _load_lf(self):
        DATASET = 'lego_figures'
        DATA_FILE = 'index.csv'

        DATA_FILE_PATH = os.path.join(self._datasets_folder, DATASET, DATA_FILE)
        DATA_FOLDER_PATH = os.path.join(self._datasets_folder, DATASET)

        df = pd.read_csv(DATA_FILE_PATH)

        data_size = df.shape[0]

        X = []
        y = []

        if self._verbose > 0:
            print("Loading data...")

        used_values = [1, 2, 15, 23]

        for i in range(data_size):
            if self._verbose > 0:
                print("Progress:", int(100 * i / data_size), "%", end='\r', flush=True)

            input_data = df.iloc[i]

            value = int(input_data['class_id'])
            if value not in used_values:
                continue

            input_image_path = os.path.join(DATA_FOLDER_PATH,
                                            input_data['path'])

            input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

            # scale down
            input_image = cv2.resize(input_image, (128, 128))

            # flip horizontally
            flipped = cv2.flip(input_image, 1)

            # flip vertically
            flipped2 = cv2.flip(input_image, 0)

            # blue
            blurred = cv2.blur(input_image, (2, 2))

            # flip and blur
            flipped_blurred = cv2.blur(flipped, (2, 2))

            input_image_vector = input_image.flatten()
            flipped_vector = flipped.flatten()
            flipped2_vector = flipped2.flatten()
            blurred_vector = blurred.flatten()
            flipped_blurred_vector = flipped_blurred.flatten()

            X.append(input_image_vector)
            X.append(flipped_vector)
            X.append(flipped2_vector)
            X.append(blurred_vector)
            X.append(flipped_blurred_vector)
            y.append(value)
            y.append(value)
            y.append(value)
            y.append(value)
            y.append(value)

        X = np.array(X)
        y = np.array(y)

        X_norm = StandardScaler().fit_transform(X)

        if self._verbose > 0:
            print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X_norm, y,
                                                            test_size=self._test_split,
                                                            random_state=42)

        if self._verbose > 0:
            print("Done!")

        return X_train, X_test, y_train, y_test

