import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

import DataHolder as dh

learning_rate=0.1
learning_rate2=0.01

training_epochs=2000
min_probability=0.80 #if < - not an object, if > - it is

#parameters for result
parametersForPredict=dh.dataToParametersForPredict(dh.loadData('parametersForPredict.json'))
predictingResults=[]

print("STARTTTTT")

def sigmodel(x):
    return 1./(1.+np.exp(-x))

#data from file
#x1_label1,x2_label1,x3_label1,x1_label2,x2_label2,x3_label2=dh.loadData('sales.json'))
data=dh.loadData('sales.json')

x1_label0=np.array([])
x2_label0=np.array([])
x3_label0=np.array([])
x1_label1=np.array([])
x2_label1=np.array([])
x3_label1=np.array([])
x1_label2=np.array([])
x2_label2=np.array([])
x3_label2=np.array([])

#data to 3 params
sales=data['sales']

for sale in sales:

    parameters=sale['parameters']

    if sale['soldObj']==0:
        x1_label0= np.append(x1_label0, parameters[0])
        x2_label0= np.append(x2_label0, parameters[1])
        x3_label0= np.append(x3_label0, parameters[2])
    elif sale['soldObj']==1:
        x1_label1= np.append(x1_label1, parameters[0])
        x2_label1= np.append(x2_label1, parameters[1])
        x3_label1= np.append(x3_label1, parameters[2])
    else:
        x1_label2= np.append(x1_label2, parameters[0])
        x2_label2= np.append(x2_label2, parameters[1])
        x3_label2= np.append(x3_label2, parameters[2])

print("parameters")
print(x1_label0)
print(x2_label0)
print(x3_label0)
print(x1_label1)
print(x2_label1)
print(x3_label1)
print(x1_label2)
print(x2_label2)
print(x3_label2)
print("parameters")    

#random data
#x1_label1=np.random.normal(3,1,1000)
#x2_label1=np.random.normal(2,1,1000)
#x3_label1=np.random.normal(4,1,1000) #third level
#x1_label2=np.random.normal(7,1,1000)
#x2_label2=np.random.normal(6,1,1000)
#x3_label2=np.random.normal(8,1,1000) #third level

#move into the another place
#x1s=np.append(x1_label1, x1_label2)
#x2s=np.append(x2_label1, x2_label2)
#x3s=np.append(x3_label1, x3_label2)#add
#ys=np.asarray([0.]*len(x1_label1)+[1.]*len(x1_label2))


X1=tf.placeholder(tf.float32, shape=(None, ), name="x1")
X2=tf.placeholder(tf.float32, shape=(None,), name="x2")
X3=tf.placeholder(tf.float32, shape=(None,), name="x3")
Y=tf.placeholder(tf.float32, shape=(None,), name="y")

w=tf.Variable([0., 0., 0., 0.], name="w", trainable=True)
w2=tf.Variable([0., 0., 0., 0.], name="w2", trainable=True) #for another classificator

y_model=tf.sigmoid(w[3]*X3+w[2]*X2+w[1]*X1+w[0])

y2_model=tf.sigmoid(w2[3]*X3+w2[2]*X2+w2[1]*X1+w2[0])  # for another model

cost=tf.reduce_mean(-tf.log(y_model*Y+(1-y_model)*(1-Y)))

cost2=tf.reduce_mean(-tf.log(y2_model*Y+(1-y2_model)*(1-Y)))  # for another model

train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
train_op2=tf.train.GradientDescentOptimizer(learning_rate2).minimize(cost2) # for another model\

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    prev_err=0
    
    x1s=np.append(x1_label0,x1_label2)
    x1s=np.append(x1s, x1_label1)
    
    x2s=np.append(x2_label0,x2_label2)
    x2s=np.append(x2s, x2_label1)
    
    x3s=np.append(x3_label0,x3_label2)
    x3s=np.append(x3s, x3_label1)
    
    ys=np.asarray([0.]*(len(x1_label0)+len(x1_label2))+[1.]*len(x1_label1))

    
    for epoch in range(training_epochs):

        err,_=sess.run([cost, train_op], {X1: x1s, X2: x2s, X3: x3s, Y: ys})

        print(epoch, err)

        if abs(prev_err-err)<0.0001:
            break

        prev_err=err
        

    #second model
    print("______second model_________")

    x1s=np.append(x1_label0,x1_label1)
    x1s=np.append(x1s, x1_label2)
    
    x2s=np.append(x2_label0,x2_label1)
    x2s=np.append(x2s, x2_label2)
    
    x3s=np.append(x3_label0,x3_label1)
    x3s=np.append(x3s, x3_label2)

    ys=np.asarray([0.]*(len(x1_label0)+len(x1_label1))+[1.]*len(x1_label2))

    prev_err=0
    
    for epoch in range(training_epochs):
                
        err,_=sess.run([cost2, train_op2], {X1: x1s, X2: x2s, X3: x3s, Y: ys})
        
        print(epoch, err)
        if abs(prev_err-err)<0.0001:
            break
        prev_err=err
        
    
    w_val=sess.run(w,{X1: x1s, X2: x2s, X3: x3s, Y: ys})
    w2_val=sess.run(w2,{X1: x1s, X2: x2s, X3: x3s, Y: ys})
    print('w: '+str(w_val)+'     ,w2: '+str(w2_val))

    #need to try without for
    #predictingResults=(sess.run(y_model, { X1:[2,1,0,1], X2:[4,2,3,7], X3:[5,3,5,6] }))
    for params in parametersForPredict:

        result1=sess.run(y_model, { X1:[params[0]], X2:[params[1]], X3:[params[2]] })[0]
        result2=sess.run(y2_model, { X1:[params[0]], X2:[params[1]], X3:[params[2]] })[0]

        if result1>result2:
            resultMax= result1
            result= 1
        else:
            resultMax=result2
            result=2
            
        
        result= 0 if resultMax<min_probability else result
       # print("resultMax:"+str(resultMax)+"  "+str(resultMax<min_probability))
        print("result: "+str(result)+" result1:"+str(result1)+" result2: "+str(result2))
        
        predictingResults.append(result)#add for multyexit

    print(str(predictingResults))
    #print('result for parameters ('+','.join([str(x1_forRes), str(x2_forRes), str(x3_forRes)])+'):'), #get and print result
    #print(sess.run(y_model, { X1:[x1_forRes], X2:[x2_forRes], X3:[x3_forRes] }))

#results to file
dh.writeDataToFile('predictedResults.json', dh.predictedResultsToJsonWithParams(parametersForPredict, predictingResults))

print('\n\n\nshow graph? 1- yessssssssss:')

if '1'!=input() :  sys.exit()
 
#x1_boundary, x2_boundary, x3_boundary=[],[], []
#for x1_test in np.linspace(0,10,100):
#    for x2_test in np.linspace(0, 10, 100):
#       for x3_test in np.linspace(0,10,100):
#           z=sigmodel(-x3_test*w_val[3]-x2_test*w_val[2]-x1_test*w_val[1]-w_val[0])
#            if abs(z-0.5)<0.01:
#                x1_boundary.append(x1_test)
#               x2_boundary.append(x2_test)
#               x3_boundary.append(x3_test)

ax=plt.axes(projection="3d")
#ax.scatter3D(x1_boundary, x2_boundary, x3_boundary, c='b', marker='o', s=20)
ax.scatter3D(x1_label0, x2_label0, x3_label0, c='b', marker='o', s=20)
ax.scatter3D(x1_label1, x2_label1, x3_label1, c='r', marker='x', s=20)
ax.scatter3D(x1_label2, x2_label2, x3_label2, c='g', marker='v', s=20)

for params in parametersForPredict:
    ax.scatter3D(params[0], params[1], params[2], c='y', marker='s', s=50)#add result points
    
plt.show()
