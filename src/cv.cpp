#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <math.h>

using namespace std;

const int input_size = 784;
const int hidden_size = 256;
const int output_size = 10;

// BMP文件头结构
#pragma pack(push, 1)
struct BMPHeader
{
    char bfType[2];  // 文件类型，必须是"BM"
    uint32_t bfSize; // 文件大小
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits; // 数据偏移量
};

// BMP信息头结构
struct BMPInfoHeader
{
    uint32_t biSize;         // 信息头大小
    int32_t biWidth;         // 图像宽度
    int32_t biHeight;        // 图像高度
    uint16_t biPlanes;       // 色彩平面数，必须是1
    uint16_t biBitCount;     // 位深度，常见的有24位、32位
    uint32_t biCompression;  // 压缩类型，0表示不压缩
    uint32_t biSizeImage;    // 图像数据大小
    int32_t biXPelsPerMeter; // 水平分辨率
    int32_t biYPelsPerMeter; // 垂直分辨率
    uint32_t biClrUsed;      // 使用的颜色数，0表示使用所有
    uint32_t biClrImportant; // 重要颜色数，0表示全部重要
};
#pragma pack(pop)

// 读取BMP文件
bool readBMP(const string &filename, vector<uint8_t> &pixelData)
{

    BMPHeader bmpHeader;
    BMPInfoHeader bmpInfoHeader;

    ifstream inputFile(filename, ios::binary);
    if (!inputFile)
    {
        cerr << "Error: Could not open file " << filename << endl;
        return false;
    }

    // 读取文件头
    inputFile.read(reinterpret_cast<char *>(&bmpHeader), sizeof(bmpHeader));
    if (bmpHeader.bfType[0] != 'B' || bmpHeader.bfType[1] != 'M')
    {
        cerr << "Error: Not a valid BMP file." << endl;
        return false;
    }

    // 读取信息头
    inputFile.read(reinterpret_cast<char *>(&bmpInfoHeader), sizeof(bmpInfoHeader));

    // 读取像素数据
    inputFile.seekg(bmpHeader.bfOffBits, ios::beg);
    int rowSize = ((bmpInfoHeader.biWidth * bmpInfoHeader.biBitCount + 31) / 32) * 4; // 每行字节数
    int imageSize = rowSize * abs(bmpInfoHeader.biHeight);
    pixelData.resize(imageSize);
    inputFile.read(reinterpret_cast<char *>(pixelData.data()), imageSize);

    inputFile.close();
    return true;
}

struct Layer
{
    vector<float> weights;
    vector<float> biases;
};

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

// 前向传播函数
struct ForwardResult
{
    vector<float> hidden;   // 隐藏层输出
    vector<float> output;   // 输出层输出
    vector<float> hidden_z; // 隐藏层加权和（激活前）
    vector<float> output_z; // 输出层加权和（激活前）
};

ForwardResult forwardPropagation(const std::vector<float> &input,
                                 const Layer &inputToHidden,
                                 const Layer &hiddenToOutput)
{
    ForwardResult result;
    result.hidden.resize(hidden_size);
    result.hidden_z.resize(hidden_size);
    result.output.resize(output_size);
    result.output_z.resize(output_size);

    // 1. 输入层到隐藏层
    for (int h = 0; h < hidden_size; h++)
    {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++)
        {
            sum += input[i] * inputToHidden.weights[i + h * input_size];
        }
        result.hidden_z[h] = sum + inputToHidden.biases[h];
        result.hidden[h] = sigmoid(result.hidden_z[h]);
    }

    // 2. 隐藏层到输出层
    for (int o = 0; o < output_size; o++)
    {
        float sum = 0.0f;
        for (int h = 0; h < hidden_size; h++)
        {
            sum += result.hidden[h] * hiddenToOutput.weights[h + o * hidden_size];
        }
        result.output_z[o] = sum + hiddenToOutput.biases[o];
        result.output[o] = sigmoid(result.output_z[o]);
    }

    return result;
}

// 反向传播函数
void backwardPropagation(const std::vector<float> &input,
                         const ForwardResult &forward_result,
                         const std::vector<float> &target,
                         Layer &inputToHidden,
                         Layer &hiddenToOutput)
{
    // 1. 计算输出层误差
    std::vector<float> output_delta(output_size);
    for (int o = 0; o < output_size; o++)
    {
        float error = forward_result.output[o] - target[o]; // 均方误差的导数
        output_delta[o] = error * sigmoid_derivative(forward_result.output_z[o]);
    }

    // 2. 计算隐藏层误差
    std::vector<float> hidden_delta(hidden_size);
    for (int h = 0; h < hidden_size; h++)
    {
        float error = 0.0f;
        for (int o = 0; o < output_size; o++)
        {
            error += output_delta[o] * hiddenToOutput.weights[h + o * hidden_size];
        }
        hidden_delta[h] = error * sigmoid_derivative(forward_result.hidden_z[h]);
    }

    // 3. 更新隐藏层到输出层的权重和偏置
    for (int o = 0; o < output_size; o++)
    {
        for (int h = 0; h < hidden_size; h++)
        {
            float grad = output_delta[o] * forward_result.hidden[h];
            hiddenToOutput.weights[h + o * hidden_size] -= learning_rate * grad;
        }
        hiddenToOutput.biases[o] -= learning_rate * output_delta[o];
    }

    // 4. 更新输入层到隐藏层的权重和偏置
    for (int h = 0; h < hidden_size; h++)
    {
        for (int i = 0; i < input_size; i++)
        {
            float grad = hidden_delta[h] * input[i];
            inputToHidden.weights[i + h * input_size] -= learning_rate * grad;
        }
        inputToHidden.biases[h] -= learning_rate * hidden_delta[h];
    }
}

int main()
{
    // 训练次数
    const int epochs = 10;

    vector<uint8_t> pixels;
    vector<float> pixelData;

    // 初始化隐藏层
    Layer inputToHidden;
    inputToHidden.weights.resize(input_size * hidden_size);
    for (auto &w : inputToHidden.weights)
    {
        w = (float)rand() / RAND_MAX * 2 - 1; // 随机初始化权重到 [-1, 1]
    }
    inputToHidden.biases.resize(hidden_size, 0.0f);

    // 初始化输出层
    Layer hiddenToOutput;
    hiddenToOutput.weights.resize(hidden_size * output_size);
    for (auto &w : hiddenToOutput.weights)
    {
        w = (float)rand() / RAND_MAX * 2 - 1;
    }
    hiddenToOutput.biases.resize(output_size, 0.0f);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int i = 0; i < 10; i++)
        {
            for (int j = 1; j <= 500; j++)
            {
                string s = to_string(j);
                string path = "../public/train_bmp/" + to_string(i) + "/" + to_string(i) + "_" + s + ".bmp";
                readBMP(path, pixels);
                for (auto pixel : pixels)
                {
                    pixelData.push_back((float)pixel / 255.0f); // 归一化到0-1
                }
                // 前向传播
                ForwardResult output = forwardPropagation(pixelData, inputToHidden, hiddenToOutput);
                // 反向传播
                backwardPropagation(pixelData, output, i,
                                    inputToHidden, hiddenToOutput);
            }
        }
    }

    return 0;
}