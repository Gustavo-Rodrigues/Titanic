import csv as csv
import numpy as np
from numpy.random import normal
import pandas as pd
import matplotlib.pyplot as plt
import math
import random


# VARIABLE DESCRIPTION:
# survival        Survival
#                 (0 = No; 1 = Yes)
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)
#
# SPECIAL NOTES:
# Pclass is a proxy for socio-economic status (SES)
#  1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
#
# Age is in Years; Fractional if Age less than One (1)
#  If the Age is Estimated, it is in the form xx.5
#
# With respect to the family relation variables (i.e. sibsp and parch)
# some relations were ignored.  The following are the definitions used
# for sibsp and parch.
#
# Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
# Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
# Parent:   Mother or Father of Passenger Aboard Titanic
# Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic
#
# Other family relatives excluded from this study include cousins,
# nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
# only with a nanny, therefore parch=0 for them.  As well, some
# travelled with very close friends or neighbors in a village, however,
# the definitions do not support such relations.


def show_data(data):
    # gaussian_numbers = normal(size=1000)
    age = data[0::,5]
    age = np.array(filter(None,age))
    plt.figure(1)
    plt.hist(age.astype(np.float),normed=1)
    plt.title("Age histogram")
    plt.xlabel("age")
    plt.ylabel("Frequency")

    # plt.show()

    sibSp = data[0::,6]
    sibSp = np.array(filter(None,sibSp))
    plt.figure(2)
    plt.hist(sibSp.astype(np.float),normed=1)
    plt.title("Siblings and spouses histogram")
    plt.xlabel("sibSp")
    plt.ylabel("Frequency")

    # plt.show()

    parch = data[0::,7]
    parch = np.array(filter(None,parch))
    plt.figure(3)
    plt.hist(parch.astype(np.float),normed=1)
    plt.title("Parents and children histogram")
    plt.xlabel("parch")
    plt.ylabel("Frequency")

    fare = data[0::,9]
    fare = np.array(filter(None,fare))
    plt.figure(4)
    plt.hist(fare.astype(np.float),normed=1)
    plt.title("Fare histogram")
    plt.xlabel("fare")
    plt.ylabel("Frequency")

    # plt.show()


    outcome = data[0::,1]

branches =  [ ['Pclass',['1','2','3'],2],
              ['Sex',['female','male'],4],
              ['Age',[10,30,50,200,''],5],
              ['SibSp',[4,10,''],6],
              ['Parch',[3,10],7],
              ['Fare',[10,50,100,''],9],
              ['Embarked',['','C','Q','S'],11]]
attrs = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# print data[0::,10]
# print data[0::,11]
def plurality_value(examples):
    positive = 0 # survived
    negative = 0 # died
    for example in examples:
        if example[1] == '1':
            positive = positive + 1
        else:
            negative = negative + 1
    if positive > negative:
        return 1
    elif positive == negative:
        return random.randint(0, 1)
    return 0


class Tree(object):
    def __init__(self):
        self.child = []
        self.attribute = None
        self.outcome = 0

    def set_outcome(self, outcome):
        self.outcome = outcome


# the structure of the data is an array with 2 columns the column 0 is the data and the columns 1 is the output
# examples is the dataset as it is, nope
# examples are the examples that we're working with
# attributes are self-evident e.g. siblings on this case
# parent_examples will be the example at it's purest form
def decision_tree_learning(examples, attributes, parent_examples, every_attr):
    # check if all examples have the same classification
    changed_state = 0
    state = examples[1]
    for example in examples:
        if example[1] != state:
            changed_state = 1
            break

    if examples is None:
        return plurality_value(parent_examples)
    elif changed_state == 0:
        return state
    elif attributes is None:
        return plurality_value(examples)
    else:
        # argmax 'a' belonging to attributes
        best_attribute = importance(examples, attributes)
        tree = Tree
        examples = []
        # find the attr
        for branch in branches:
            if branch[0] == best_attribute:
                # for each attr do everything again
                for attr in branch[1]:
                    examples = [x for x in examples[0::,branch[2]] if x]
                    # TODO a for equal to the data one, including the append
            break




def importance(examples, attributes, every_attr):
    # for each attribute
    # for each  ~ decision ~
    # for attr in attributes:
    # pag 704
    best_attr = ''
    biggest_gain = 0
    for attribute in range(0,len(attributes)):
        # find the position on the dataset
        for pos,attr in enumerate(every_attr):
            if attr == attributes[attribute]:
                position = pos
                attribute = 0
                gain = 0
                # data[0::,1] = examples[0::, 1]
                data = []
                for x in range( len(examples[0::,position])):
                    data.append([examples[x,position],examples[x,1]])
                data = np.array(data)


                # Pclass ok
                # Sex ok
                # Age ok
                # SibSp ok
                # Parch ok
                # Fare ok
                # Embarked ok


                if attr == 'Age':
                    gain = information_gain(data,[10,30,50,200,''],1)
                elif attr == 'SibSp':
                    gain = information_gain(data,[4,10,''],1)
                elif attr == 'Parch':
                    gain = information_gain(data, [3, 10], 1)
                elif attr == 'Fare':
                    gain = information_gain(data, [10, 50, 100, ''], 1)
                else:
                    unique_values = np.unique(data[0::, 0])
                    gain = information_gain(data,unique_values,0)
                if gain > biggest_gain:
                    biggest_gain = gain
                    best_attr = attr
                break
    return best_attr


def information_gain(data,split_values,ranged):
    # so basically we need for each split value calculate the entropy
    # reference russel norvig ai a modern approach 2nd edition page 660
    # p = positives
    # n = negatives
    # the i is the ith element
    # the remainder is the sum of the p+n/total*entropy(pi/pi+ni,ni/pi+ni)
    remainder = []
    total_positives = 0
    for value in range(len(split_values)):
        # count the total of occurences of that value and how many of them were positive
        occurrences = 0
        positives = 0
        for i in range(len(data)):
            if ranged:
                if value > 0 and split_values[value] != '' and data[i,0] != '':
                    # print  value,float(data[i, 0])
                    if split_values[value-1] < float(data[i,0]) and split_values[value] >= float(data[i,0]):
                        occurrences += 1
                        # print 'HERE'
                        if data[i,1] == '1':
                            positives +=  1
                            total_positives += 1
                else:
                    if split_values[value] != '' and data[i,0] != '':
                        if split_values[value] >= float(data[i,0]):
                            occurrences += 1
                            if data[i,1] == '1':
                                positives +=  1
                                total_positives += 1
                    else:
                        if split_values[value] == data[i,0]:
                            occurrences += 1
                            if data[i,1] == '1':
                                positives +=  1
                                total_positives += 1

            else:
                if split_values[value] == data[i,0]:
                    occurrences += 1
                    if data[i,1] == '1':
                        positives +=  1
                        total_positives += 1
        p = float(occurrences)/float(len(data))
        if p == 0:
            remainder.append(0)
        else:
            remainder.append( p*entropy(occurrences,positives) )
    # print 'Remainder:', remainder
    info_gain = entropy(len(data),total_positives) - sum(remainder)
    # print 'Information gain:', info_gain
    return info_gain

def entropy(occurrences, positives):
    p = float(positives)/float(occurrences)
    # print 'P:',p,'positives',positives,'occurrences',occurrences
    if p == 0:
        return 0
    if p == 1:
        return 0
    delta = -((p) * math.log(p, 2) + (1 - p) * math.log(1 - p, 2))
    # print 'Entropy:',delta
    return delta


def main():
    csv_file = csv.reader(open('/home/gustavo/Documents/Kaggle/Titanic/Data/train.csv', 'rb'))
    header = csv_file.next()
    print
    print 'FIELDS:'
    print header, '\n'
    attributes = header
    print 'Quantity of passengers:'

    # read the file and creates the array
    data = []
    for row in csv_file:
        data.append(row)
    data = np.array(data)
    passengers = np.size(data[0::, 1].astype(np.float))
    print passengers, '\n'

    print 'Example of data:'
    print data[0], '\n'
    # print data

    # let's find out how many survived
    survivors = np.sum(data[0::, 1].astype(np.float))
    print 'Survivors:'
    print survivors
    print 'Proportion of survivors:'
    print survivors / passengers, '\n'

    test = importance(data, attrs, header)
    print test
    # print branches

main()
