import tensorflow as tf
import numpy as np
import csv

input_x = []
f1 = csv.reader(open('TrainSamples.csv','r'))
for index, i in enumerate(f1):
    temp_x = []
    for elm in range(len(i)):
        temp_x.append(int(float(i[elm])))
    input_x.insert(index, temp_x)

output_y = []
f2 = csv.reader(open('TrainLabels.csv','r'))
for index, i in enumerate(f2):
    temp_x = []
    for elm in range(len(i)):
        temp_x.append(int(float(i[elm])))
    input_label=[]
    for k in range(10):
        if(k != temp_x[0]):
            input_label.append(0)
        else:
            input_label.append(1)
    output_y.insert(index, input_label)
    if(index < 10):
        print(input_label)

inputs = tf.keras.Input(shape=(84,))
x = tf.keras.layers.Dense(169, use_bias=True, activation='sigmoid')(inputs)
x_1 = tf.keras.layers.Dense(169, use_bias=True, activation='sigmoid')(x)
outputs = tf.keras.layers.Dense(10, use_bias=True, activation='sigmoid')(x_1)
m = tf.keras.Model(inputs, outputs)     # 使用 输入 和 输出 创建模型

m.compile(tf.keras.optimizers.SGD(learning_rate=0.1), 'mse')
m.fit(input_x, output_y, epochs=1000, batch_size=1, verbose=1)

# 保存模型
m.save('widow_model.h5')