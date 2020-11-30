import tensorflow as tf
import numpy as np
import csv

test_case = []
f1 = csv.reader(open('TrainSamples.csv','r'))
for index, i in enumerate(f1):
    temp_x = []
    for elm in range(len(i)):
        temp_x.append(int(float(i[elm])))
    test_case.insert(index, temp_x)

# 恢复模型
restored_model = tf.keras.models.load_model('widow_model.h5')

f2 = open("Result.csv", 'w', newline="")
writer2 = csv.writer(f2)

for i in range(len(test_case)):
    temp_case = np.array([test_case[i]])
    temp_case = (restored_model.predict(temp_case)).ravel();
    flag = False;
    for i in range(len(temp_case)):
        if abs(temp_case[i] - 1) < 0.4 :
            writer2.writerow([i])
            flag = True
            break;
    if not flag :
        writer2.writerow([-1])

print("Finish!")