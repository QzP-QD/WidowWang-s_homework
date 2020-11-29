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
    if(index < 10):
        print(temp_x)

print("-----------")

output_y = []
f2 = csv.reader(open('TrainLabels.csv.csv','r'))
for index, i in enumerate(f2):
    temp_x = []
    for elm in range(len(i)):
        temp_x.append(int(float(i[elm])))
    # print(len(temp_x))
    output_y.insert(index, temp_x)

inputs = tf.keras.Input(shape=(84,))
x = tf.keras.layers.Dense(169, use_bias=True, activation='sigmoid')(inputs)
outputs = tf.keras.layers.Dense(1, use_bias=True, activation='sigmoid')(x)
m = tf.keras.Model(inputs, outputs)     # 使用 输入 和 输出 创建模型

m.compile(tf.keras.optimizers.SGD(learning_rate=0.1), 'mse')
m.fit(input_x, output_y, epochs=1000, batch_size=1, verbose=0)

# 保存模型
m.save('widow_model.h5')