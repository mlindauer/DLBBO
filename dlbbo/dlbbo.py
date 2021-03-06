import glob
import os
import typing
import logging

import scipy
import numpy as np

import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD
import keras.applications as app_models
from keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dlbbo.scenario.aslib_scenario import ASlibScenarioDL

def as_loss(y_true, y_pred):
            return K.dot(y_true, K.transpose(y_pred))

class DLBBO(object):

    def __init__(self, scenario_dn: str, 
                 model_type: str="VGG-like",
                 verbose:int=1):
        ''' Constructor '''

        self.scenario_dn = scenario_dn
        self.logger = logging.getLogger("DLBBO")

        self.IMAGE_SIZE = 64  # pixel
        self.EPOCHS = 1000

        self.model_type = model_type
        self.verbose = verbose
        self.loss = "categorical_crossentropy"
        self.loss = "as_loss"

    def main(self):
        '''
        main method
        '''

        scenario, inst_image_map = self.read_data()

        sum_qual = 0
        sbs_idx = np.argmin(scenario.performance_data.sum(axis=0).values)
        y_sel = []
        y_sbs = []
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
            for p in y_pred:  # several images for each test instance:
                print(p)
                pred_algo_idx = np.argmax(p)
                qual = scenario.performance_data.ix[test_inst, pred_algo_idx]
                sbs_qual = scenario.performance_data.ix[test_inst, sbs_idx]
                print(qual, sbs_qual)
                y_sel.append(qual)
                y_sbs.append(sbs_qual)
        self.logger.info("Average Quality: %f" % (np.average(y_sel)))
        vbs = scenario.performance_data.min(axis=1).mean()
        sbs = scenario.performance_data.mean(axis=0).min()
        self.logger.info("VBS: %f" % (vbs))
        self.logger.info("SBS: %f" % (sbs))
        y_sbs = np.array(y_sbs)
        y_sel = np.array(y_sel)
        self.scatter_plot(y_sbs, y_sel)

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
                image = scipy.misc.imread(img_fn)
                # resize
                image = scipy.misc.imresize(
                    image, size=(self.IMAGE_SIZE, self.IMAGE_SIZE))
                # scale
                image = image / 256  # 256bit grey scale
                inst_image_map[inst].append(image)  # scale to [0,1]

        return scenario, inst_image_map

    def build_train_data(self, scenario: ASlibScenarioDL, inst_image_map: typing.Dict, test_inst: str):
        '''
            generate X,y from scenario data
        '''

        X_train, X_test = [], []
        y_train, y_test = [], []
        n_classes = len(scenario.algorithms)
        for inst in scenario.instances:
            for image in inst_image_map[inst]:
                #X = np.reshape(image, (self.IMAGE_SIZE, self.IMAGE_SIZE, 1))
                X = image
                perfs = scenario.performance_data.loc[inst].values
                #y = np.argmin(perfs)
                if inst != test_inst:
                    X_train.append(X)
                    y_train.append(perfs)
                else:
                    X_test.append(X)
                    y_test.append(perfs)

        return np.array(X_train), np.array(y_train), \
            np.array(X_test), np.array(y_test), \
            n_classes
        
    def preprocess_y(self, y, n_classes):
        '''
            preprocess y for different loss functions
        '''
        
        if self.loss == "categorical_crossentropy":
            y = np.argmin(y, axis=1)
            y = keras.utils.to_categorical(y, num_classes=n_classes)
        elif self.loss == "as_loss" or self.loss == as_loss:
            self.EPOCHS = 100
            y = y - np.repeat([np.min(y, axis=1)], y.shape[1], axis=0).T
            y /= np.max(y)
            self.loss = as_loss
        else:
            raise ValueError("Unkonwn loss function")
            
        return y
            

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_classes: int):
        '''
            train model, VGG-like network
        '''
        
        y_train = self.preprocess_y(y_train, n_classes)
        y_test = self.preprocess_y(y_test, n_classes)

        if self.model_type == "VGG-like":
            model = self.get_vgg_model(n_classes=n_classes)
        else:
            
            if self.model_type == "MobileNet":
                self.EPOCHS = 100
                model = app_models.mobilenet.MobileNet(input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3), 
                                                   include_top=False, 
                                                   weights=None, 
                                                   classes=n_classes,
                                                   dropout=1e-2,
                                                   alpha=0.5)
            elif self.model_type == "VGG16":
                self.EPOCHS = 100
                model = app_models.vgg16.VGG16(input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3), 
                                                   include_top=False, 
                                                   weights=None, 
                                                   classes=n_classes)
            else:
                raise ValueError("Unknown model_type")
            
            x = model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(64, activation='relu')(x)
            pred_layer = Dense(n_classes, activation='softmax')(x)
            model = Model(inputs=model.input, outputs=pred_layer)
            sgd = SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True)
            model.compile(optimizer=sgd, loss=self.loss)

        model.fit(X_train, y_train, batch_size=32, epochs=self.EPOCHS,
                  verbose=self.verbose)
        train_score = model.evaluate(X_train, y_train, batch_size=32)
        test_score = model.evaluate(X_test, y_test, batch_size=32)

        print(train_score, test_score)
        y = model.predict(X_test)

        return y

    def get_vgg_model(self, n_classes: int):

        model = Sequential()
        # input: 128x128 images with 1 channel -> (128, 128, 1) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        model.add(Conv2D(32, (3, 3), activation='relu',
                         input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax'))

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=self.loss, optimizer=sgd)

        return model

    def scatter_plot(self, x, y):

        max_v = np.max((x, y))
        min_v = np.min((x, y))
        ax = plt.subplot(111)
        ax.scatter(x, y)
        ax.set_xlabel('Single Best')
        ax.set_ylabel('DNN')
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([min_v, max_v])
        ax.set_ylim([min_v, max_v])

        plt.tight_layout()
        plt.savefig("scatter_%s.png" %(self.model_type))
