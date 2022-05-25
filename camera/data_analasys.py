import numpy as np
import math

data = np.genfromtxt('data/results/6/results60.csv', delimiter='\n')

print(data)

data_app = []
array_app = []
ind = 0
hit_nan = 0
for i in range(len(data)):

    if math.isnan(data[i]) == True:

        if hit_nan == 1:

            hit_nan = 2

        if hit_nan == 2:
            for j in range(ind+1,i):
                array_app.append(data[j])
            data_app.append(array_app)

            array_app = []
            hit_nan = 0
            ind = 0

        if ind == 0:
            ind = i

        hit_nan = 1


        # print(i,'i')
        # print(hit_nan,'hit_nan')
        # print(ind,'ind')
        # print(array_app,'array_app')
        # print(data_app,'data_app')

# print(data_app,'data_app')
# print(np.shape(data_app),'data_app')

error_sphere = []
error_cyliner = []
surface_class = []
surface_class_eig = []
first_run = 1
last_i = 0

for i in range(len(data_app)):
    # print(i)

    if first_run == 0:
        i = last_i+len(data_app[last_i])+4
    first_run = 0

    last_i = i
    # print(i)
    if i >= len(data_app):
        break

    len_i = len(data_app[i])

    error_sphere.append(data_app[i])
    error_cyliner.append(data_app[i+1])
    surface_class.append(data_app[i+2])
    surface_class_eig.append(data_app[i+3])


    # print(error_sphere,'error_sphere')
    # print(i)
    # input()

# print('###########################################################')
print(error_sphere,'error_sphere')
print(error_cyliner,'error_cyliner')
print(surface_class,'surface_class')
print(surface_class_eig,'surface_class_eig')

print(len(error_sphere[0]),'error_sphere')
# print((error_sphere),'error_sphere')

#0 sphere 1 clyinder 2 error
ground_truth = np.array([[2,1,0,1,1,1,0],[1,0,2,1,1,1,1],[2,2,2,1,1,1,1,0],\
                         [1,1,1,1,0,1,2,2],[0,1,1,1,1,1,2,2]])

TP = 0
TN = 0
FP = 0
FN = 0
total = 0


for i in range(5):
    print(i,'i')

    total += len(error_sphere[i])

    for j in range(len(error_sphere[i])):

        print(j,'j')

        if error_sphere[i][j] > error_cyliner[i][j]:
            #True possitive cylinder
            if ground_truth[i][j] == 1:
                TP += 1
                print(TP,'TP')
            if ground_truth[i][j] == 0:
                FP += 1
                print(FP,'FP')
        else:
            #TP sphere
            if ground_truth[i][j] == 0:
                TP += 1
                print(TP,'TP2')
            #TP sphere
            if ground_truth[i][j] == 1:
                FP += 1
                print(FP,'FP2')

accuracy = (TP+TN)/total

print()


precision = TP/(TP+FP)

recall = TP/(TP+FN)

#fals positive rate
FPR = FP/(TN+FP)

F1_score = (2 * precision * recall)/(precision+recall)

print(total,'total')
print(TP,'TP')
print(FP,'FP')
print(TN,'TN')
print(FN,'FN')
print(accuracy,'accuracy')
print(precision,'precision')
print(recall,'recall')
print(FPR,'FPR')
print(F1_score,'F1_score')
