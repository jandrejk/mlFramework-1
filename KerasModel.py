import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser
import numpy as np
from pandas import DataFrame,concat
import json
np.random.seed(0)
import conf.keras_models as keras_models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model as lm
from keras.utils.np_utils import to_categorical
from collections import deque
import time
import os
from sklearn import model_selection

class KerasObject():

    def __init__(self, parameter_file = "", variables=[], target_names = {}, filename = "", balancedbatches = "False", classes = [""], era = ''):

        self.variables = variables
        self.models = []
        self.balancedbatches = balancedbatches
        self.classes = classes
        self.era = era

        try:
            if filename: self.load(filename)
            elif not parameter_file or not variables:
                raise Warning("Warning! Object not defined. Load from file or set 'params' and 'variables'")
            else:
                with open(parameter_file,"r") as FSO:
                    params = json.load(FSO)
                self.params = params["model"]
        except Warning as e:
            print e
            self.params = []

        if target_names: self.target_names = target_names



    def load(self, filename):
        with open(filename + ".dict", 'rb') as FSO:
            tmp_dict = json.load(FSO)

        print "Loading model from: " + filename 
        self.__dict__.clear()
        self.__dict__.update(tmp_dict)

        self.models = []
        for model in tmp_dict["models"]:
            self.models.append( lm(model) )

    def save(self, filename):
        placeholders = []
        tmp_models = []
        for i,model in enumerate(self.models):
            modelname = filename + ".fold{0}".format(i)
            model.save( modelname )
            tmp_models.append(model)
            placeholders.append( modelname )
        self.models = placeholders

        with open(filename + ".dict", 'wb') as FSO:
            json.dump(self.__dict__, FSO)

        self.models = tmp_models


    def train(self, samples):

        if type(samples) is list:
            samples = deque(samples)

        for i in xrange( len(samples) ):
            test = samples[0]
            train = [ samples[1] ]

            for j in xrange(2, len(samples) ):
                train.append( samples[j] )
            
            train = concat(train , ignore_index=True).reset_index(drop=True)

            self.models.append( self.trainSingle( train, test ) )
            samples.rotate(-1)

        print "Finished training!"


    def trainSingle(self, train, test):

        # writing targets in keras readable shape
        best = str(int(time.time()))
        y_test  = to_categorical( test["target"].values )

        N_classes = len(self.classes)

        model_impl = getattr(keras_models, self.params["name"]) # reads model defined in conf/keras_models.py
        model = model_impl(len(self.variables), N_classes)
        model.summary()

        if (self.balancedbatches=="True"):

            print "---------------------------------Balanced batches are used --------------------------------"
            x_train, x_val, y_train, y_val, w_train, w_val = model_selection.train_test_split(
                                                             train[self.variables].values,
                                                             train["target"].values,
                                                             train["train_weight"].values,
                                                             test_size=0.25,
                                                             random_state=1234)

            x_train_df = DataFrame(x_train, columns=train[self.variables].columns)
            x_val_df = DataFrame(x_val, columns=train[self.variables].columns)

            eras=[]
            if self.era == 'Run2' :
                eras=["era2016","era2017","era2018"]
            else :
                eras.append("era"+self.era)

            eraIndexDict = {
                era: {
                    label: np.where((x_train_df[self.variables][era] == 1) & (y_train == i_class))[0]
                    for i_class, label in enumerate(self.classes)
                     }
                for i_era, era in enumerate(eras)
                }

            def balancedBatchGenerator(eventsPerClassAndBatch):
                    while True:
                        nperClass = int(eventsPerClassAndBatch)
                        selIdxDict = {
                            era: {
                                label: eraIndexDict[era][label][np.random.randint(
                                    0, len(eraIndexDict[era][label]), nperClass)]
                                for label in self.classes
                            }
                            for era in eras
                        }
                        y_collect = to_categorical(np.concatenate([
                            y_train[selIdxDict[era][label]] for label in self.classes
                            for era in eras
                        ]))
                        x_collect = np.concatenate([
                            x_train[selIdxDict[era][label], :] for label in self.classes
                            for era in eras
                        ])
                        w_collect = np.concatenate([
                            w_train[selIdxDict[era][label]] * (eventsPerClassAndBatch*len(self.classes) / np.sum(w_train[selIdxDict[era][label]])) for label in self.classes
                            for era in eras
                        ])
                        yield x_collect, y_collect, w_collect

            testIndexDict = {
                era: {
                    label: np.where((x_val_df[self.variables][era] == 1) & (y_val == i_class))[0]
                    for i_class, label in enumerate(self.classes)
                    }
                for i_era, era in enumerate(eras)
            }

            def calculateValidationWeights(x_val, y_val, w_val):
                y_collect = to_categorical(np.concatenate([
                    y_val[testIndexDict[era][label]] for label in self.classes
                    for era in eras
                    ]))
                x_collect = np.concatenate([
                    x_val[testIndexDict[era][label], :] for label in self.classes
                    for era in eras
                    ])
                w_collect = np.concatenate([
                    w_val[testIndexDict[era][class_]] * (len(x_val) / np.sum(w_val[testIndexDict[era][class_]]))
                    for class_ in self.classes
                    for era in eras
                    ])
                return x_collect, y_collect, w_collect

            x_val, y_val, w_val = calculateValidationWeights(x_val, y_val, w_val)

            history = model.fit_generator(
                    balancedBatchGenerator(self.params["eventsPerClassAndBatch"]),
                    steps_per_epoch=self.params["steps_per_epoch"]*len(eras),
                    epochs=self.params["epochs"],
                    callbacks=[EarlyStopping(patience=self.params["early_stopping"]), ModelCheckpoint( best + ".model", save_best_only=True, verbose = 1) ],
                    validation_data=(x_val, y_val, w_val)
            )
        else :
            print "---------------------------------Balanced batches are NOT used --------------------------------"
            y_train = to_categorical( train["target"].values )
            history = model.fit(
                train[self.variables].values,
                y_train,
                sample_weight=train["train_weight"].values,
                validation_split = 0.25,
                # validation_data=(test[self.variables].values, y_test, test["train_weight"].values),
                batch_size=self.params["batch_size"],
                epochs=self.params["epochs"],
                shuffle=True,
                callbacks=[EarlyStopping(patience=self.params["early_stopping"]), ModelCheckpoint( best + ".model", save_best_only=True, verbose = 1) ])

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt

        print "plotting training"
        epochs = xrange(1, len(history.history["loss"]) + 1)
        plt.plot(epochs, history.history["loss"], lw=3, label="Training loss")
        plt.plot(epochs, history.history["val_loss"], lw=3, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        if not os.path.exists("plots"):
            os.mkdir("plots")
        plt.savefig("plots/fold_{0}_loss.png".format(best), bbox_inches="tight")


        print "Reloading best model"
        model = lm(best + ".model")
        os.remove( best + ".model" )

        return model

    def predict(self, samples, where=""):

        predictions = []
        if type(samples) is list:
            samples = deque(samples)

        for i in xrange( len(samples) ):
            predictions.append( self.testSingle( samples[0], i ) )
            samples.rotate(-1)

        samples[0].drop(samples[0].index, inplace = True)
        samples[1].drop(samples[1].index, inplace = True)

        return predictions


    def testSingle(self, test,fold ):

        prediction = DataFrame( self.models[fold].predict(test[self.variables].values) )

        return DataFrame(dtype = float, data = {"predicted_class":prediction.idxmax(axis=1).values,
                                 "predicted_prob": prediction.max(axis=1).values } )
