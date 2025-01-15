#include <MicroTFLite.h>
#include "model.h" // 导入模型数据

// 定义用于存储中间张量的内存区域
constexpr int kTensorArenaSize = 2 * 1024; // 根据模型大小调整
alignas(16) uint8_t tensorArena[kTensorArenaSize];

void setup() {
  // 初始化串口通信
  Serial.begin(9600);

  // 初始化模型
  if (!ModelInit(model, tensorArena, kTensorArenaSize)) {
    Serial.println("Model initialization failed!");
    while (true); // 如果初始化失败，停止执行
  }

  Serial.println("Model initialization done.");
  ModelPrintMetadata();
  ModelPrintTensorInfo();
  ModelPrintInputTensorDimensions(); // 打印输入张量的维度信息
  ModelPrintOutputTensorDimensions(); // 打印输出张量的维度信息
}

void loop() {
  float input_value = 3.0; // 测试输入值

  // 显示输入值
  Serial.print("输入值：");
  Serial.println(input_value);

  // 设置输入值到模型
  if (!ModelSetInput(input_value, 0)) { // 设置第一个输入
    Serial.println("Failed to set input value!");
    return;
  }

  // 运行推理
  if (!ModelRunInference()) {
    Serial.println("Inference failed!");
    return;
  }

  // 获取模型的输出
  float prediction = ModelGetOutput(0);

  // 打印输出结果（预测的平方值）
  Serial.print("预测输出：");
  Serial.println(prediction);

  delay(2000); // 每2秒运行一次
}
