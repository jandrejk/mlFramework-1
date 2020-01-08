from Reader import Reader
import json
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--era', help='Era (2016, 2017, 2018 or Run2) to use' ,choices = ['2016','2017','2018','Run2'],  default = '2016')
    parser.add_argument('-c', '--channel', nargs='+', help='define channels to use in a space seperated list, e.g. -c mt tt' ,choices = ['mt','et','tt','em'])
    parser.add_argument('-measure', '--measurement', help='targeted type of measurement', choices = ['inclusive', 'stage0', 'stage1p2'])

    args = parser.parse_args()

    era = args.era
    channel_list = args.channel
    measurement = args.measurement

    print 'considered channel(s) : ' + str(channel_list)

    for channel in channel_list:
        config_to_use = "conf/global_config_{0}_{1}.json".format(channel,era)
        print "config : "  + config_to_use
        train_weights = {}
        train_weights[channel] = getWeights(config_to_use, channel, era, measurement)

        class_weights = {}
        for cl in ["ztt", "zll", "misc", "tt", "w", "ss", "noniso", "ggh", "qqh","diboson","singletop", "signal", "ggh_PTHGT200", "ggh_0J", "ggh_1J_PTH0to120", "ggh_1J_PTH120to200", "ggh_2J", "vbftopo_lowmjj", "vbftopo_highmjj", "qqh_2J", "qqh_PTHGT200"]:
            class_weights[cl] = {}
            for ch in ["mt","et","tt","em"]:
                tmp = train_weights.get(ch,{})
                class_weights[cl][ch] = tmp.get(cl,0)


    for cl in class_weights:
        print '"{0}":'.format(cl) + " "*(7-len(cl)),'{'+'"mt":{mt}, "et":{et}, "tt":{tt}, "em":{em} '.format(**class_weights[cl])+'},'

def getWeights(samples, channel, era, measurement):

    train_weights = {}
    read = Reader(channel = channel,
                  era = era,
                  config_file = samples,
                  folds=2,
                  measurement=measurement)

    numbers = {}
    targets = []
    

    for sample, sampleName in read.get(what = "nominal"):
        target =  read.config["target_names"][ sample[0]["target"].iloc[0] ]

        numbers[ sampleName["histname"] ] = {"total": len(sample[0]) + len(sample[1]) }
        numbers[ sampleName["histname"] ]["evt"] = sample[0]["event_weight"].sum() + sample[1]["event_weight"].sum()


        if numbers.get(target, None) == None:
            numbers[target] = {"evt": sample[0]["event_weight"].sum() + sample[1]["event_weight"].sum() }
            numbers[target]["total"] = len(sample[0]) + len(sample[1])
            targets.append( target )
        else:
            numbers[target]["evt"] += sample[0]["event_weight"].sum() + sample[1]["event_weight"].sum()
            numbers[target]["total"] += len(sample[0]) + len(sample[1])
        

    total = 0.
    for n in numbers:
        if n in targets: continue
        total += numbers[n]["evt"]
    # print total
    for n in numbers:
        if n not in targets: continue
        train_weights[n] = round(total / numbers[n]["evt"], 1)

        print n+" "*(15 - len(n)) ,round(total / numbers[n]["evt"], 1)

    return train_weights



if __name__ == '__main__':
    main()
