import random
import re
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.linear_model import LogisticRegression
import urllib
from sklearn.svm import SVC
import csv
import subprocess
import os
from heapq import nlargest


def initiatePopulation(PeptideAminoNumber):
    population = []
    aminos = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    initialPopulationLength = 100
    for i in range(int(initialPopulationLength)):  # number of the individuals in the initial population
        peptide = [aminos[random.randint(0, 19)] for i in range(int(PeptideAminoNumber))]
        peptide_last = ''.join(peptide)
        population.append(peptide_last)
        # print (population)
    return population

def unkownFeatures(path):
    df = pd.read_excel(path)
    UnknownFeatures = df[['mdx_age', 'IV_dose(mg/Kg)', 'num_of_IVinject', 'IM']]
    uniqueUnknownFeatures = UnknownFeatures.drop_duplicates()
    return uniqueUnknownFeatures


def prepareInputForTool(population):
    input_list = []
    m = 0
    for i in population:
        input_text = ">" + str(m) + "\n" + i + "\n"
        m += 1
        input_list.append(input_text)

    return input_list

def getC2PredResults():
    C2Pred_values = []
    with open('result.txt', 'r') as resultFile:
        results = resultFile.read()
        results = results.replace('&nbsp;', '')
        results = results.replace('&nbsp', '')
        results = results.replace('<font color=red><strong>', '')
        results = results.replace('</strong', '')
        results = results.replace('</font>', '')
        results = results.split('\n')
        for item in results:
            if (item[0:4] == 'CPP>'):
                C2Pred_values.append(item[4:12])
            elif (item[0:8] == 'Non-CPP>'):
                C2Pred_values.append(item[8:16])
    #with open('result.txt', 'w') as resultFile:
        #resultFile.truncate(0)
    return C2Pred_values

def createDataframe(population, c2Pred_value, uniqueUnknownFeatures):
    peptides = pd.DataFrame(columns=['sequence', 'length', '#R', 'mdx_age', 'IV_dose(mg/Kg)', 'num_of_IVinject', 'IM'])

    for p in population:
        for i in range(len(uniqueUnknownFeatures.index)):
            peptides = peptides.append({'sequence': p, 'length': len(p), '#R': p.count('R'),
                                        'mdx_age': uniqueUnknownFeatures['mdx_age'].iloc[i],
                                        'IV_dose(mg/Kg)': uniqueUnknownFeatures['IV_dose(mg/Kg)'].iloc[i],
                                        'num_of_IVinject': uniqueUnknownFeatures['num_of_IVinject'].iloc[i],
                                        'IM': uniqueUnknownFeatures['IM'].iloc[i]}, ignore_index=True)

    new_value_c2Pred_dataframe = pd.DataFrame({'C2Pred': c2Pred_value})
    peptideListComplete = peptides.join(new_value_c2Pred_dataframe)

    peptideListFinal = peptideListComplete.drop(['sequence'], axis=1)

    #C2Pred_dataframe = peptideListComplete[['sequence', 'C2Pred']]
    #peptideListFinal = peptideListFinal.drop_duplicates()
    #peptideListComplete = peptideListComplete.drop_duplicates()
    return peptideListFinal, peptideListComplete

def prepareInitialDataset(path):
    df = pd.read_excel(path)
    y_train = df[['Class']]
    y_test = y_train[33:45]
    y_train = y_train[:33]
    X_train = df.drop(['Class', 'cpp_name', 'sequence', 'Western_Blot_TAmuscle', 'CPPred'], axis=1)
    X_test = X_train.iloc[33:45]
    X_train = X_train[:33]
    return (X_train, X_test, y_train, y_test)

def modelTraining(model, X_train, y_train):
    trainedModel = model.fit(X_train, y_train.values.ravel())
    return trainedModel


def modelPedictProba(trainedModel, peptide_list):
    #print ('peptide list ', peptide_list)
    probabilities = trainedModel.predict_proba(peptide_list)
    return probabilities


def write_proba_to_file(probabilities):
    with open('probabilities.csv', mode='w') as probabfile:
        proba_writer = csv.writer(probabfile, delimiter=',')

        for row in probabilities:
            proba_writer.writerow(row)


def sortResults(probabilities, population, next_generation, peptideDataset, topSelectedResultsList):
    results = []
    dist = 0
    if (len(next_generation)) == 0:
        #print ('generation 0')
        return results
    for (p, index) in zip(probabilities, range(len(probabilities))):
        if p[1] > 0.5:
            for i in range(len(next_generation)):
                dist = 0
                for k in range(len(population)):
                    dist = dist + (hamming_distance(population[k], next_generation[i]) * 0.001)

                        #print(population[k], next_generation[i], hamming_distance(population[k], next_generation[i]))
            topSelectedResultsList = topSelectedResultsList.append(peptideDataset.iloc[index], ignore_index = True)
            results.append((index, ((p[1] + dist),(len(topSelectedResultsList) - 1))))
    #print ('results', results)
    return results, topSelectedResultsList


def selectMutationType():
    # 1 delete, 2 add, 3 change
    mutationType = np.random.choice(np.arange(1, 4), p=[0.20, 0.20, 0.60])
    return mutationType


def remove_duplicates(next_generation):
  return list(dict.fromkeys(next_generation))


def createNextGeneration(peptideDataset, PeptideAminoNumber, population, mutationType):
    next_generation = []
    aminos = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    randPlace = random.randrange(0, PeptideAminoNumber)
    if (mutationType == 1):  # delete an amino acid from sequence
        for i in population:
            newstr = i[:randPlace] + i[randPlace + 1:]
            i = newstr
            next_generation.append(i)
    elif (mutationType == 2):  # add an amino acid in sequence
        randamino = aminos[random.randint(0, 19)]
        for i in population:
            newstr = i[:randPlace] + randamino + i[randPlace:]
            i = newstr
            next_generation.append(i)
    else:  # change an aminoacid
        randamino = aminos[random.randint(0, 19)]
        for i in population:
            newstr = i[:randPlace] + randamino + i[randPlace + 1:]
            i = newstr
            next_generation.append(i)
    next_generation =  remove_duplicates(next_generation)
    return next_generation

def hamming_distance(peptide1, peptide2):
    return sum(amino1 != amino2 for amino1, amino2 in zip(peptide1, peptide2))

def search_in_csv(csv_name, input_list):
    csv_list = pd.read_csv(csv_name, header=None)
    have_values = []
    if (not csv_list.empty):
        csv_list = csv_list.drop_duplicates()
        index = -1
        for instance in input_list:
            for index in range(0, len(csv_list)):
                if instance == csv_list[0].iloc[index]:
                    frame = csv_list.iloc[index]
                    have_values.append(frame.values.tolist())
    return have_values


def remove_have_values(have_values_list, population):
    population_for_tool = population
    values_from_csv = []
    removed_pept = []
    for item in have_values_list:
        for indiv in population_for_tool:
            if item[0] == indiv:
                population_for_tool.remove(indiv)
                values_from_csv.append(item[1])
                removed_pept.append(indiv)
    return (population_for_tool, values_from_csv, removed_pept)

def C2Pred_tool_values(population):
    c2Pred_value = []
    c2Pred_final = []
    input_list = prepareInputForTool(population)
    with open('1.txt', 'w') as inputFile:
        inputFile.truncate(0)
        for item in input_list:
            inputFile.write("%s" % item)

    os.system("predict.exe")
    c2Pred_value = c2Pred_value + getC2PredResults()
    #print ('c2Pred_value', c2Pred_value)
    for value in c2Pred_value:
        for i in range(17):
            c2Pred_final.append(value)
    return population, c2Pred_final

def get_key(val, dialogi_previous):
    for key, value in dialogi_previous.items():
         if val == value:
             return key

def select_best_results(top_results_nextGen, top_resultsDict, topSelectedResultsList, dialogi_previous):
    top_results_nextDict = dict(top_results_nextGen)
    dialogi_next = {}
    if (topSelectedResultsList.empty):
        return
    for (key_nextDict, value_next) in top_results_nextDict.items():
        if key_nextDict in [key for (key, value) in top_resultsDict.items()]:
            if value_next[0] < top_resultsDict[key_nextDict][0]:
                valueInOldList = (top_resultsDict[key_nextDict])[1]
                seq = topSelectedResultsList['sequence'].iloc[valueInOldList]
                dialogi_next[seq] = top_resultsDict[key_nextDict][0]
            else:
                seq = topSelectedResultsList['sequence'].iloc[value_next[1]]
                dialogi_next[seq] = top_results_nextDict[key_nextDict][0]
        else:
            seq = topSelectedResultsList['sequence'].iloc[value_next[1]]
            dialogi_next[seq] = value_next[0]

    if len(dialogi_previous) > 0:
        for (k, v) in dialogi_previous.items():
            if k not in [k1 for k1, v1 in dialogi_next.items()]:
                seq = get_key(v, dialogi_previous)
                dialogi_next[seq] = v
    #print ('top_resultsDict', top_resultsDict)
    #print ('dialogi_previous', dialogi_previous)
    #print ('dialogi_next', dialogi_next)
    return dialogi_next, top_results_nextGen


def main():
    peptide = []
    dialogi = {}
    path = "DMD_dataset.xls"
    PeptideAminoNumber = 14  # input("Give Peptide length: ")
    pept = [14]
    for p in pept:
        print ('amino number', p)
        PeptideAminoNumber = p
        population = initiatePopulation(PeptideAminoNumber)
        population, c2Pred_value = C2Pred_tool_values(population)
        uniqueUnknownFeatures = unkownFeatures(path)

        peptideDataset, peptideListComplete = createDataframe(population, c2Pred_value,
                                                              uniqueUnknownFeatures)

        X_train, X_test, y_train, y_test = prepareInitialDataset(path)
        model = LogisticRegression()
        trainedModel = modelTraining(model, X_train, y_train)
        probabilities = modelPedictProba(trainedModel, peptideDataset)
        write_proba_to_file(probabilities)

        next_generation = []
        print("initial population peptides ", population)
        #print ('initial probabilities', probabilities)
        topSelectedResultsList = pd.DataFrame(columns=['sequence', 'length', '#R', 'mdx_age', 'IV_dose(mg/Kg)', 'num_of_IVinject', 'IM', 'C2Pred'])
        top_results, topSelectedResultsList = sortResults(probabilities, population, population, peptideListComplete, topSelectedResultsList)
        top_resultsDict = dict(top_results)
        for i in range(50):
            if (top_results) and (population):
                mutationType = selectMutationType()

                next_generation = createNextGeneration(population, PeptideAminoNumber, population, mutationType)

                print("next_generation peptides ", next_generation)
                #print("previous generation ", top_results)

                population, c2Pred_value = C2Pred_tool_values(population)

                peptideNextGeneration, peptideListNextComplete = createDataframe(next_generation,
                                                                                 c2Pred_value, uniqueUnknownFeatures)
                #print ('peptideNextGeneration', peptideNextGeneration)
                if (not peptideNextGeneration.empty):
                    probabilities = modelPedictProba(trainedModel, peptideNextGeneration)

                    #print ('probabilities round ', i)
                    #print ('probabilities results', probabilities)

                    top_results_nextGen, topSelectedResultsList = sortResults(probabilities, next_generation, population, peptideListNextComplete, topSelectedResultsList)
                # print "next generation results ", top_results_nextGen

                    dialogi_next, top_results_nextGen = select_best_results(top_results_nextGen, top_resultsDict, topSelectedResultsList, dialogi)
                    #print ('dialogi', dialogi)
                    dialogi = dialogi_next
                    population = next_generation
                    top_results = top_results_nextGen
                    top_resultsDict = dict(top_results)
                peptideListComplete = peptideListNextComplete

            # print ("peptideListComplete ", peptideListComplete)
        if len(dialogi) > 0:
            dialogi_sorted = nlargest(10, dialogi, key = dialogi.get)
            print("dialogi ", dialogi_sorted)
            print("length dialogi ", len(dialogi_sorted))
        else:
            print("no results")
    return


if __name__ == '__main__':
    main()