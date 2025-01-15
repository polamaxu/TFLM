"""
环境要求：
- Python 3.6+
- TensorFlow 2.x
- NumPy

功能说明：
这个程序演示了一个简单的回归模型，用于预测数字的平方值。
包含模型训练、保存、转换为TFLite格式，以及生成C数组三个主要步骤。
"""

import tensorflow as tf
import numpy as np

# 1. 准备训练数据
# 创建一个简单的数据集：输入为1到5的数字，输出为其平方值
x_train = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
y_train = np.square(x_train)  # 计算平方值作为目标输出

# 2. 定义模型
# 使用Sequential API创建一个简单的模型
# Lambda层直接实现了平方运算，无需训练参数
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), name='input_layer'),
    tf.keras.layers.Lambda(lambda x: x ** 2, name='square_layer')
])

# 3. 编译模型
# 注：由于使用Lambda层直接计算平方，实际上不需要训练
model.compile(optimizer='sgd', loss='mean_squared_error')

# 4. 测试模型性能
print("测试结果：")
for x in x_train:
    y_pred = model.predict([x])
    print(f"输入：{x[0]}, 预测输出：{y_pred[0][0]}")

# 5. 保存模型
# 将模型保存为HDF5格式
model.save('./saved_model.h5')

# 6. 转换为TFLite格式
# TFLite格式更适合在嵌入式设备上运行
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存TFLite模型
with open('./model.tflite', 'wb') as f:
    f.write(tflite_model)

# 7. 将TFLite模型转换为C数组
def convert_tflite_to_c_array(tflite_model_path, c_file_path):
    """
    将TFLite模型文件转换为C语言数组格式
    
    参数:
        tflite_model_path: TFLite模型文件路径
        c_file_path: 输出的C头文件路径
    """
    with open(tflite_model_path, 'rb') as file:
        content = file.read()

    with open(c_file_path, 'w') as file:
        file.write('const unsigned char model_data[] = {')
        for i, val in enumerate(content):
            if i % 12 == 0:  # 每12个数字换行，提高可读性
                file.write('\n  ')
            file.write('0x{:02x}, '.format(val))
        file.write('\n};\n')
        file.write('const int model_data_len = {};\n'.format(len(content)))

if __name__ == '__main__':
    convert_tflite_to_c_array('./model.tflite', './model_data.h')