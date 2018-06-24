print('loading')
import pandas as pd
from sklearn import svm
import csv

def listRevLook(lookedUp, arrayy):
    for num, char in enumerate(arrayy,start=1):
        if lookedUp.lower == char.lower:
            return num
    return False

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           'á', 'â', 'æ', 'ä', 'ç', 'é', 'è', 'ê', 'í', 'î', 'ñ', 'ó', 'ô', 'ö', 'ú', 'ù', 'û', 'ü']
letters2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           'a1', 'a2', 'a3', 'a4', 'c1', 'e1', 'e2', 'e3', 'i1', 'i2', 'n1', 'o1', 'o2', 'o3', 'u1', 'u2', 'u3', 'u4']
languageIndex = ['english' ,'spanish', 'french', 'italian']
langDict = {'english':0 ,'spanish':1, 'french':2, 'italian':3}
def findLetterFreq(fileName):
    letterFreq = [0] * len(letters)
    totalLetters = 0
    try:
        file = open(fileName, "r")
        text=file.read()
        file.close()
    except:
        text=fileName
    for letter in text:
        index = listRevLook(letter, letters)
        if index:
            totalLetters += 1
            letterFreq[index - 1] += 1
    i = 0
    p = 100 / totalLetters
    while i < len(letterFreq):
        letterFreq[i] *= p
        i += 1
    return letterFreq

def makeTypeLabel(lang):
    datapoints=lang['Type']
    langType = [0]*len(datapoints)
    for i in range(0,len(datapoints)):
        langType[i]=langDict[datapoints[i]]
    return langType

def addTrainingFile(lang, fileName):
    with open('languages.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow([lang.lower()]+findLetterFreq(fileName))

languages=pd.read_csv("languages.csv")
letterMat=languages[letters2].as_matrix()
type_label = makeTypeLabel(languages)
language_features = languages.columns.values[1:].tolist()
model = svm.SVC(kernel='linear', decision_function_shape='ovr')
model.fit(letterMat, type_label)
print('loading done \n')

def distinguishLang(textFile):
    print('This text is in '+languageIndex[model.predict([findLetterFreq(textFile)])[0]])

distinguishLang("this is in english")
distinguishLang("c'est en français")
distinguishLang('esto es en español')
distinguishLang("questo è in italiano")
distinguishLang('test.txt')