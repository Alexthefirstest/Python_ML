import json

#'sales.json'
def loadData(filename):

    with open (filename, "r") as readFile:
        data=json.load(readFile)

    return data


def dataToParameters(data):

    x1_label1=[]
    x2_label1=[]
    x3_label1=[]
    x1_label2=[]
    x2_label2=[]
    x3_label2=[]

    arraysSet0=[x1_label1, x2_label1, x3_label1]
    arraysSet1=[x1_label2, x2_label2, x3_label2]

    sales=data['sales']

    for sale in sales:

        parameters=sale['parameters']
        arraysToInsert= arraysSet0 if sale['soldObj'] == 0 else arraysSet1
        
        arraysToInsert[0].append(parameters[0])
        arraysToInsert[1].append(parameters[1])
        arraysToInsert[2].append(parameters[2])

    return x1_label1, x2_label1, x3_label1, x1_label2, x2_label2, x3_label2


def dataToParametersForPredict(data):
    return data['parametersForPredict']


def writeDataToFile(filename, data):

    with open (filename, "w") as writeFile:
        
        #json.dump(str(data), writeFile)
        writeFile.write(data)

    return None


def predictedResultsToJsonWithParams(parameters, predictedValues):

    jsonResult=['{"predictedResults":[\n']
    
    i=0

    for pv in predictedValues:

        jsonResult.append('{"parameters":')
        jsonResult.append(str(parameters[i]))
        jsonResult.append(', "predict":')
        jsonResult.append("%.3f" % pv+'}')
        jsonResult.append(',\n')
        i=i+1

    jsonResult.pop()
    jsonResult.append('\n]}')
    return ''.join(jsonResult)


"""params=dataToParametersForPredict(loadData('parametersForPredict.json'))
print(params)
print('\n\n\n')
results=[1,2,3,4]
print(results)
print('\n\n\n')
pr=predictedResultsToJsonWithParams(params, results)
print( pr )
writeDataToFile('predictedResults.json', pr)
"""
"""x1, x2, x3, y1, y2, y3=dataToParameters(loadData('sales.json'))

print(x1)
print(x2)
print(x3)
print(y1)
print(y2)
print(y3)

print(loadData('parametersForPredict.json'))
print(dataToParametersForPredict(loadData('parametersForPredict.json')))"""
