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
    flag = False
    max_index = (temp_case.tolist() ).index(max(temp_case, key = abs))
    writer2.writerow([max_index])

print("Finish!")