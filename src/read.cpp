#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <string>
using namespace std;

const int input_size = 784;
const int hidden_size = 256;
const int output_size = 10;

// BMP文件头结构
#pragma pack(push, 1)
struct BMPHeader
{
    char bfType[2];
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader
{
    uint32_t biSize;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t biXPelsPerMeter;
    int32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
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

    inputFile.read(reinterpret_cast<char *>(&bmpHeader), sizeof(bmpHeader));
    if (bmpHeader.bfType[0] != 'B' || bmpHeader.bfType[1] != 'M')
    {
        cerr << "Error: Not a valid BMP file." << endl;
        return false;
    }

    inputFile.read(reinterpret_cast<char *>(&bmpInfoHeader), sizeof(bmpInfoHeader));

    inputFile.seekg(bmpHeader.bfOffBits, ios::beg);
    int rowSize = ((bmpInfoHeader.biWidth * bmpInfoHeader.biBitCount + 31) / 32) * 4;
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

// 前向传播函数（简化版，仅需输出）
vector<float> forwardPropagation(const vector<float> &input,
                                 const Layer &inputToHidden,
                                 const Layer &hiddenToOutput)
{
    vector<float> hidden(hidden_size);
    for (int h = 0; h < hidden_size; h++)
    {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++)
        {
            sum += input[i] * inputToHidden.weights[i + h * input_size];
        }
        hidden[h] = sigmoid(sum + inputToHidden.biases[h]);
    }

    vector<float> output(output_size);
    for (int o = 0; o < output_size; o++)
    {
        float sum = 0.0f;
        for (int h = 0; h < hidden_size; h++)
        {
            sum += hidden[h] * hiddenToOutput.weights[h + o * hidden_size];
        }
        output[o] = sigmoid(sum + hiddenToOutput.biases[o]);
    }

    return output;
}

// 加载模型
bool loadModel(Layer &inputToHidden, Layer &hiddenToOutput, const string &filename)
{
    ifstream inFile(filename, ios::binary);
    if (!inFile)
    {
        cerr << "Error: Could not open file " << filename << " for reading." << endl;
        return false;
    }

    // 读取输入到隐藏层的权重和偏置
    uint32_t weights_size, biases_size;
    inFile.read(reinterpret_cast<char *>(&weights_size), sizeof(weights_size));
    inFile.read(reinterpret_cast<char *>(&biases_size), sizeof(biases_size));
    inputToHidden.weights.resize(weights_size);
    inputToHidden.biases.resize(biases_size);
    inFile.read(reinterpret_cast<char *>(inputToHidden.weights.data()), weights_size * sizeof(float));
    inFile.read(reinterpret_cast<char *>(inputToHidden.biases.data()), biases_size * sizeof(float));

    // 读取隐藏到输出层的权重和偏置
    inFile.read(reinterpret_cast<char *>(&weights_size), sizeof(weights_size));
    inFile.read(reinterpret_cast<char *>(&biases_size), sizeof(biases_size));
    hiddenToOutput.weights.resize(weights_size);
    hiddenToOutput.biases.resize(biases_size);
    inFile.read(reinterpret_cast<char *>(hiddenToOutput.weights.data()), weights_size * sizeof(float));
    inFile.read(reinterpret_cast<char *>(hiddenToOutput.biases.data()), biases_size * sizeof(float));

    inFile.close();
    cout << "Model loaded from " << filename << endl;
    return true;
}

// 找到输出中最大值的索引
int getPredictedDigit(const vector<float> &output)
{
    int max_index = 0;
    float max_value = output[0];
    for (int i = 1; i < output_size; i++)
    {
        if (output[i] > max_value)
        {
            max_value = output[i];
            max_index = i;
        }
    }
    return max_index;
}

int main()
{
    // 加载模型
    Layer inputToHidden, hiddenToOutput;
    if (!loadModel(inputToHidden, hiddenToOutput, "model.bin"))
    {
        return 1;
    }

    // 读取测试图像
    string image_path = "test.bmp";

    vector<uint8_t> pixels;
    vector<float> pixelData;

    for (int i = 0; i < 10; i++)
    {
        for (int j = 1; j <= 500; j++)
        {
            string s = to_string(j);
            string path = "../public/train_bmp/" + to_string(i) + "/" +
                          to_string(i) + "_" + s + ".bmp";

            pixels.clear();
            pixelData.clear();
            if (!readBMP(path, pixels))
            {
                continue;
            }
            for (auto pixel : pixels)
            {
                pixelData.push_back((float)pixel / 255.0f);
            }
            // 进行预测
            vector<float> output = forwardPropagation(pixelData, inputToHidden, hiddenToOutput);
            int predicted_digit = getPredictedDigit(output);
            // 输出结果
            cout << "Predicted digit: " << predicted_digit << endl;
        }
    }

    return 0;
}