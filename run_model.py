from Reader import Reader
import copy
import pandas
import json
import sys
import os
from glob import glob
import argparse
import cPickle
import subprocess as sp
import multiprocessing as mp
import keras

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='channel', help='Decay channel' ,choices = ['mt','et','tt','em'], default = 'mt')
    parser.add_argument('-m', dest='model',   help='ML model to use' ,choices = ['keras','xgb'],  default = 'keras')
    parser.add_argument('-t', dest='train',   help='Train new model' , action='store_true')
    parser.add_argument('-s', dest='short',   help='Do !!NOT!! predict shapes' , action='store_true')
    parser.add_argument('-d', dest='datacard',  help='Only produce Datacard' , action='store_true')
    parser.add_argument('-e', dest='era',  help='Era' , choices=["2016","2017", "2018", "Run2"], required = True)
    parser.add_argument('--add_nominal', dest='add_nom',  help='Add nominal samples to prediction', action='store_true' )
    parser.add_argument('-measure', dest='measurement', help='Targeted type of measurement', choices = ['inclusive', 'stage0', 'stage1p2'], default = 'stage0')
    parser.add_argument('-b', dest='balancedbatches', help='Use balanced batches', choices = ["True", "False"], default = "False")
    args = parser.parse_args()

    print "---------------------------"
    print "Era: ", args.era
    print "Running over {0} samples".format(args.channel)
    print "Using {0}".format(args.model), keras.__version__
    if args.train:
        print "Training new model"
    if args.short:
        print "Not predicting shape templates."
    print "Targeted type of measurement : ", args.measurement
    print "Using balcanced batches : ", args.balancedbatches
    print "---------------------------"


    run(samples = "conf/global_config_{0}_{1}.json".format(args.channel,args.era),
        channel=args.channel,
        era = args.era,
        use = args.model,
        train = args.train,
        short = args.short,
        measurement = args.measurement,
        datacard = args.datacard,
        add_nominal = args.add_nom,
        balancedbatches = args.balancedbatches)

def run(samples,channel, era, use, train,short, measurement, datacard = False, add_nominal=False, balancedbatches = "False"):

    # import model
    if use == "xgb":
        from XGBModel import XGBObject as modelObject
        parameters = "conf/parameters_xgb.json"

    if use == "keras":
        from KerasModel import KerasObject as modelObject
        parameters = "conf/parameters_keras.json"

    # create instance of Reader
    read = Reader(channel = channel,
                  config_file = samples,
                  folds=2,
                  era = era,
                  measurement = measurement,
                  balancedbatches= balancedbatches)

    target_names = read.config["target_names"]
    variables = read.config["variables"]

    models_folder = era + "/models"
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    modelname = "{0}/{1}.{2}".format(models_folder,channel,use)
    scaler = None

    if train:
        print "Training new model"
        print "Loading Training set"
        # load samples for training. This returns the input files merged (ggH+qqH+...), randomized and split into two folds
        trainSet = read.getSamplesForTraining()

        # use StandardScaler from sklearn and "train" it on the training set. Then pickle it store it for later and finaly apply it to the training set.
        print "Fit Scaler to training set...",
        scaler = trainScaler(trainSet, variables )

        print " done. Dumping for later."
        with open("{0}/StandardScaler.{1}.pkl".format(models_folder,channel), 'wb') as FSO:
            cPickle.dump(scaler, FSO , 2)
        scaler = [scaler, scaler] # Hotfix since KIT uses 2 scalers

        trainSet = applyScaler(scaler, trainSet, variables)

        classes = read.getClasses()
        # load model and start training
        model = modelObject( parameter_file = parameters,
                             variables =variables,
                             target_names = target_names,
                             balancedbatches = balancedbatches,
                             classes = classes,
                             era=era)
        model.train( trainSet ) # done in KerasModel.py
        model.save(modelname)

    # prediction starts
    #load scaler and trained model
    elif not datacard:
        # TODO: Maybe not needed to check. Just load what is there
        if os.path.exists("{0}/StandardScaler.{1}.pkl".format(models_folder,channel) ):
            print "Loading Scaler"
            scaler = []
            if glob("{0}/{1}_*_keras_preprocessing.pickle".format(models_folder,channel)) :
                with open( "{0}/{1}_fold0_keras_preprocessing.pickle".format(models_folder,channel), "rb" ) as FSO:
                    scaler.append( cPickle.load( FSO ) )

                with open( "{0}/{1}_fold1_keras_preprocessing.pickle".format(models_folder,channel), "rb" ) as FSO:
                    scaler.append( cPickle.load( FSO ) )
            else:
                with open( "{0}/StandardScaler.{1}.pkl".format(models_folder,channel), "rb" ) as FSO:
                    tmp = cPickle.load( FSO )
                    scaler = [tmp,tmp]


        print "Loading model and predicting."
        model = modelObject( filename = modelname )
        read.variables = model.variables
        variables = model.variables

    if not datacard:

        outpath = read.config["outpath"] + "/predictions_" + era
        predictions = {}
        print "Predicting samples"
        # only used by DESY
        if add_nominal:
            print "Predicting Nominal"
            for sample, sampleConfig in read.get(what = "nominal", for_prediction = True):
                sandbox(channel, model, scaler, sample, variables, "nom_" + sampleConfig["histname"], outpath ,sampleConfig, read.modifyDF )

        # Iterate over nominal samples
        for sample, sampleConfig in read.get(what = "full", add_jec = not short, for_prediction = True):
            if "data" in sampleConfig["histname"]:
                sandbox(channel, model, scaler, sample, variables, "NOMINAL_ntuple_Data", outpath, sampleConfig, read.modifyDF)
            elif "full" in sampleConfig["histname"]:
                sandbox(channel, model, scaler, sample, variables,  "NOMINAL_ntuple_" + sampleConfig["histname"].split("_")[0], outpath, sampleConfig, read.modifyDF )
            else:
                splName = sampleConfig["histname"].split("_")
                sandbox(channel, model, scaler, sample, variables,  "_".join(splName[1:])+"_ntuple_" + sampleConfig["histname"].split("_")[0], outpath, sampleConfig, read.modifyDF )
        # shape files
        if not short:
            print "Predicting shapes"
            for sample, sampleConfig in read.get(what = "tes", for_prediction = True):
                sandbox(channel, model, scaler, sample, variables, sampleConfig["histname"], outpath ,sampleConfig, read.modifyDF )



    if "hephy.at" in os.environ["HOME"]:
        from Tools.Datacard.produce import Datacard, makePlot
        from Tools.CutObject.CutObject import Cut
        from Tools.FakeFactor.FakeFactor import FakeFactor

        Datacard.use_config = era + "/datacard_conf"
        D = Datacard(channel=channel,
                     variable="predicted_prob",
                     era=era,
                     real_est="mc",
                     add_systematics = not short,
                     debug=True,
                     use_cutfile = "conf/cuts_{0}.json".format(era))

        FakeFactor.fractions = "{0}/datacard_conf/fractions/htt_ff_fractions_{0}.root".format(era) 

        D.create(era+"/"+use)
        makePlot(channel, "ML", era+"/"+use, era, era+"/plots")

# run prediction for each sample
def sandbox(channel, model, scaler, sample, variables, outname, outpath, config = None, modify = None):
    # needed because of memory management
    # iterate over chunks of sample and do splitting on the fly
    first = True
    for part in sample:
        # This is awful. Try to figure out a better way to add stuff to generator.
        if modify:
            modify(part, config)

        part["THU"] = 1 # Add dummy
        # Carefull!! Check if splitting is done the same for training. This is the KIT splitting
        folds = [part.query( "abs(evt % 2) != 0 " ).reset_index(drop=True), part.query( "abs(evt % 2) == 0 " ).reset_index(drop=True) ]
        addPrediction(channel, model.predict( applyScaler(scaler, folds, variables) ), folds, outname, outpath, new = first )
        
        folds[0].drop(folds[0].index, inplace=True)
        folds[1].drop(folds[1].index, inplace=True)
        part.drop(part.index, inplace=True)

        first = False
    del sample

def addPrediction(channel,prediction, df, sample, outpath, new = True):

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    for i in xrange( len(df) ):
        for c in prediction[i].columns.values.tolist():
            df[i][c] =  prediction[i][c]
            
        if i == 0 and new: mode = "w"
        else: mode = "a"
        # df[i].to_root("{0}/{1}-{2}.root".format("predictions",channel, sample), key="TauCheck", mode = mode)
        df[i].to_root("{0}/{1}-{2}.root".format(outpath,channel, sample), key="TauCheck", mode = mode)
        prediction[i].drop(prediction[i].index, inplace = True)

def trainScaler(folds, variables):
    from sklearn.preprocessing import StandardScaler

    total = pandas.concat( folds, ignore_index = True ).reset_index(drop=True)
    print " ...... Variables 'era2016', 'era2017', 'era2018' removed for scaling!"
    variables_toscale = [var for var in variables if var not in ('era2016','era2017','era2018')]
    Scaler = StandardScaler()
    Scaler.fit( total[ variables_toscale ] ) # computes the mean and std to be used for later scaling

    return Scaler

def applyScaler(scaler, folds, variables):
    if not scaler: return folds
    newFolds = copy.deepcopy(folds)
    for i,fold in enumerate(newFolds):
        variables_toscale = [var for var in variables if var not in ('era2016','era2017','era2018')]
        fold[variables_toscale] = scaler[i].transform( fold[variables_toscale] ) # perform standardization by centering and scaling
    return newFolds


if __name__ == '__main__':
    main()
