#Bayes Classifier
from __future__ import print_function
#import tensorflow as tf
from PIL import Image
import numpy as np
import PIL

I1 = np.asarray(PIL.Image.open('1.gif'))
I2 = np.asarray(PIL.Image.open('2.gif'))
I3 = np.asarray(PIL.Image.open('3.gif'))
I4 = np.asarray(PIL.Image.open('4.gif'))
#100 Non-river sample points
X_nr = np.asarray([0,10,20,30,40,50,60,70,80,90,100,149,139,202,212,225,240,260,280,299,55,298,280,275,400,420,435,440,450,460,480,490,499,505,510,416,425,281,219,382,479,204,229,183,60,67,100,422,394,306,350,380,399,412,422,445,465,485,498,511,256,268,294,315,415,265,500,415,496,428,425,183,197,194,189,224,209,231,142,64,185,146,84,72,218,166,279,171,213,440,247,238,266,213,199,273,237,168,185,414])
Y_nr = np.asarray([2,10,59,65,75,85,95,84,86,95,103,101,121,202,213,220,50,234,265,298,320,340,360,380,400,235,495,355,289,212,105,155,185,120,199,187,222,214,169,169,225,188,228,307,346,376,400,62,89,138,306,142,252,363,323,212,256,321,365,400,450,420,410,465,485,412,176,398,256,456,215,176,213,299,338,51,404,365,166,274,18,83,367,230,406,379,174,310,203,175,369,389,387,383,409,387,391,320,337,394]) 
#50 River sample points
X_r = np.asarray([160,160,160,168,169,186,185,188,186,187,202,192,197,199,176,176,172,172,166,165,170,219,212,186,176,174,173,179,212,222,176,157,152,166,175,229,161,174,172,173,179,192,214,230,206,217,200,191,185,184])
Y_r = np.asarray([19,27,49,85,85,411,415,406,395,402,267,258,258,262,230,225,215,224,196,178,114,321,289,415,87,148,125,99,354,311,129,7,23,32,57,326,125,150,184,220,428,386,346,322,332,337,356,391,416,386])
#Mean for non-river sample points
m1=0
m2=0
m3=0
m4=0
for i in range(100): #
    m1 = m1 + I1[X_nr[i]][Y_nr[i]]
    m2 = m2 + I2[X_nr[i]][Y_nr[i]]
    m3 = m3 + I3[X_nr[i]][Y_nr[i]]
    m4 = m4 + I4[X_nr[i]][Y_nr[i]]
m1 = m1/100
m2 = m2/100
m3 = m3/100
m4 = m4/100
T1 = np.asarray([m1,m2,m3,m4])
#Mean for river sample points
mr1=0
mr2=0
mr3=0
mr4=0
for i in range(50): #
    mr1 = mr1 + I1[X_r[i]][Y_r[i]]
    mr2 = mr2 + I2[X_r[i]][Y_r[i]]
    mr3 = mr3 + I3[X_r[i]][Y_r[i]]
    mr4 = mr4 + I4[X_r[i]][Y_r[i]]
mr1 = mr1/50
mr2 = mr2/50
mr3 = mr3/50
mr4 = mr4/50
T2 = np.asarray([mr1,mr2,mr3,mr4])
I = [[[0 for k in range(512)] for j in range(512)] for i in range(4)]
for j in range(512):
    for k in range(512):
        I[0][j][k] = I1[j][k]
for j in range(512):
    for k in range(512):
        I[1][j][k] = I2[j][k]
for j in range(512):
    for k in range(512):
        I[2][j][k] = I3[j][k]
for j in range(512):
    for k in range(512):
        I[3][j][k] = I4[j][k]
Cv_R  = [[0 for k in range(4)] for j in range(4)]
for i in range(4):
    for j in range(4):
        for k in range(50):
            Cv_R[i][j] = Cv_R[i][j] + (I[i][X_r[k]][Y_r[k]] - T2[i])*(I[j][X_r[k]][Y_r[k]] - T2[j])
        Cv_R[i][j] = Cv_R[i][j]/50
Cv_NR = [[0 for k in range(4)] for j in range(4)]
for i in range(4):
    for j in range(4):
        for k in range(100):
            Cv_NR[i][j] = Cv_NR[i][j] + (I[i][X_nr[k]][Y_nr[k]] - T1[i])*(I[j][X_nr[k]][Y_nr[k]] - T1[j])
        Cv_NR[i][j] = Cv_NR[i][j]/100
River_class     = [[0 for k in range(512)] for j in range(512)]
Non_River_class = [[0 for k in range(512)] for j in range(512)]
for i in range(512):
    for j in range(512):
        Test_data = np.asarray([I[0][i][j],I[1][i][j],I[2][i][j],I[3][i][j]])
        River_class[i][j] =np.dot(np.dot(np.transpose(np.subtract(Test_data, T2)) ,np.linalg.inv(Cv_R)),np.subtract(Test_data, T2))
for i in range(512):
    for j in range(512):
        Test_data = np.asarray([I[0][i][j],I[1][i][j],I[2][i][j],I[3][i][j]])
        Non_River_class[i][j] =np.dot(np.dot(np.transpose(np.subtract(Test_data, T1)) ,np.linalg.inv(Cv_NR)),np.subtract(Test_data, T1))
DRC = np.linalg.det(Cv_R)
DNRC= np.linalg.det(Cv_NR)
p1 = [[0 for k in range(512)] for j in range(512)]
p2 = [[0 for k in range(512)] for j in range(512)]
for i in range(512):
    for j in range(512):
        p1[i][j] = (-0.5)*(1/np.sqrt(DRC))*np.exp(River_class[i][j])
for i in range(512):
    for j in range(512):
        p2[i][j] = (-0.5)*(1/np.sqrt(DNRC))*np.exp(Non_River_class[i][j])
P1 = 0.3
P2 = 0.7
data = np.zeros((512, 512, 3), dtype=np.uint8)
for i in range(512):
    for j in range(512):
        if (P1 * p1[i][j]) >= (P2 * p2[i][j]):
            data[i][j] = 0
        else:
            data[i][j] = 255
img = Image.fromarray(data, 'RGB')
img.save('my4.png')
img.show()