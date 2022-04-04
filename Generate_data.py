import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Flatten

fname = 'ethylene_CO.txt'
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
# print(header)
# print(len(lines))

import numpy as np
float_data = np.zeros((len(lines), 16))
d=0
for d, line in enumerate(lines):
    p=[]
    for x in line.split(' '):
        if x=='':
            continue
        p.append(x)
    if len(p)==0:
        continue
    values = [float(x) for x in p[3:]]
    float_data[d, :] = values


print(float_data[0])
size=len(float_data)
training_size=int(2*size/3)
validation_size=int(1/10*size)
testing_size=(size-training_size)-validation_size
print("training size:",training_size)
print("testing size:",testing_size)
print("validation size:",validation_size)

print(training_size+testing_size)
print(size-training_size-testing_size)


#normalization2
mi=float_data.min(axis=0)
ma=float_data.max(axis=0)
float_data-=mi
float_data/=(ma-mi)




#data generator
# def dgenerator(data, dimensionarray, delay, min_index, max_index, batch_size, darray,shuffle=False):
#     if max_index is None:
#         max_index = len(data) - delay - 1
#     samples = np.empty((0, sum(dimensionarray)), float)
#     targets=np.empty((0,16),float)
#     max=0
#     for sensor in range(0,16):
#         dimension=dimensionarray[sensor]
#         print(sensor)
#         d=darray[sensor]
#         if max < (dimension*d):
#             max=dimension*d
#
#     i=min_index+max
#     while 1:
#         if i>=max_index:
#             break
#         i=i+1
#         ta=[]
#         sa=[]
#         for sensor in range(0,16):
#             dimension = dimensionarray[sensor]
#             d = darray[sensor]
#             c=i
#             for step in range(0,dimension):
#                 c-=d
#                 sa.append(data[c][sensor])
#         ta.append(data[i])
#         print(data[i])
#         print(np.array(ta).shape)
#         print(targets.shape)
#         samples= np.append(samples, np.array([sa]),axis=0)
#         targets=np.append(targets,np.array(ta),axis=0)
#
#     np.save("train_x_methane.npy", samples)
#     np.save("train_y_methane.npy", targets)
#     return samples,targets


def dgenerator(data, dimensionarray, delay, min_index, max_index, batch_size, darray, shuffle=False):
    if max_index is None:
        max_index = len(data) - delay - 1
    samples = np.empty((0, batch_size, sum(dimensionarray)), float)
    targets = np.empty((0, batch_size, 16), float)
    max = 0
    for sensor in range(0, 16):
        dimension = dimensionarray[sensor]

        d = darray[sensor]
        if max < (dimension * d):
            max = dimension * d

    i = min_index + max
    while 1:
        if i + batch_size >= max_index:
            break
        rows = np.arange(i, min(i + batch_size, max_index))

        target = []
        sample = []
        for b in range(0, len(rows)):
            i = i + 1

            sa = []
            for sensor in range(0, 16):
                dimension = dimensionarray[sensor]
                d = darray[sensor]
                c = i
                for step in range(0, dimension):
                    c -= d
                    sa.append(data[c][sensor])

            sample.append(sa)
            target.append(data[i])

        print(np.array(target).shape)
        print(targets.shape)
        print(np.array(sample).shape)
        print(samples.shape)
        samples = np.append(samples, np.array([sample]), axis=0)
        targets = np.append(targets, np.array([target]), axis=0)

    np.save("test_x_CO.npy", samples)
    np.save("test_y_CO.npy", targets)
    return samples, targets


m_methane= [20,19,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
d_methane = [6960,1,8000,8000,7000,7000,7000,4760,5080,7000,7000,7000,7000,7000,7000,7000]
delay = 1

m_co=[20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
d_co=[6950,2500,15000,15000,11880,4830,14000,14000,7420,8650,6000,7000,3830,4480,16000,15000]

dgenerator(float_data,m_co,delay,testing_size,None,128,d_co)

