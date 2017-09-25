import glob
import os

import scipy
import numpy as np

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from dlbbo.scenario.aslib_scenario import ASlibScenarioDL


class DLBBO(object):

    def __init__(self, scenario_dn: str):
        ''' Constructor '''

        self.scenario_dn = scenario_dn

    def main(self):
        '''
        main method
        '''

        scenario, inst_image_map = self.read_data()

        for test_inst in scenario.instances:
            print(test_inst)
            X_train, y_train, X_test, y_test, n_classes = \
                self.build_train_data(
                    scenario=scenario,
                    inst_image_map=inst_image_map,
                    test_inst=test_inst)
            y_pred = self.train(X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test,
                                n_classes=n_classes)
            print(y_pred)

    def read_data(self):
        '''
            read all scenario files
            and find all images

            Returns
            -------
            scenario: ASlibScenarioDL
        '''

        scenario = ASlibScenarioDL()
        scenario.read_scenario(dn=self.scenario_dn)

        # find images
        inst_image_map = {}
        for inst in scenario.instances:
            inst_image_map[inst] = []
            for img_fn in glob.glob(os.path.join(self.scenario_dn, "images", inst + "*")):
                inst_image_map[inst].append(scipy.misc.imread(
                    img_fn, flatten=True) / 256)  # scale to [0,1]

        return scenario, inst_image_map

    def build_train_data(self, scenario, inst_image_map, test_inst):
        '''
            generate X,y from scenario data
        '''

        X_train, X_test = [], []
        y_train, y_test = [], []
        n_classes = len(scenario.algorithms)
        for inst in scenario.instances:
            for image in inst_image_map[inst]:
                if inst != test_inst:
                    X_train.append(np.reshape(image, (128, 128, 1)))
                    perfs = scenario.performance_data.loc[inst].values
                    y_train.append(np.argmin(perfs))
                else:
                    X_test.append(np.reshape(image, (128, 128, 1)))
                    perfs = scenario.performance_data.loc[inst].values
                    y_test.append(np.argmin(perfs))

        return np.array(X_train), np.array(y_train), \
            np.array(X_test), np.array(y_test), \
            n_classes

    def train(self, X_train, y_train,
              X_test, y_test,
              n_classes):
        '''
            train model
        '''

        y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes=n_classes)

        model = Sequential()
        # input: 128x128 images with 1 channel -> (128, 128, 1) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        model.add(Conv2D(32, (3, 3), activation='relu',
                         input_shape=(128, 128, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd, metrics=['accuracy'])

        model.fit(X_train, y_train, batch_size=32, epochs=42)
        train_score = model.evaluate(X_train, y_train, batch_size=32)
        test_score = model.evaluate(X_test, y_test, batch_size=32)

        print(train_score, test_score)
        y = model.predict(X_test)

        return y
