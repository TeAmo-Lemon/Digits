#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <string>
#include <random>
using namespace std;

const int input_size = 784;
const int hidden_size = 256;
const int output_size = 10;
const float learning_rate = 0.01f;

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

float sigmoid_derivative(float x)
{
    float sig = sigmoid(x);
    return sig * (1.0f - sig);
}

struct ForwardResult
{
    vector<float> hidden;
    vector<float> output;
    vector<float> hidden_z;
    vector<float> output_z;
};

ForwardResult forwardPropagation(const vector<float> &input,
                                 const Layer &inputToHidden,
                                 const Layer &hiddenToOutput)
{
    ForwardResult result;
    result.hidden.resize(hidden_size);
    result.hidden_z.resize(hidden_size);
    result.output.resize(output_size);
    result.output_z.resize(output_size);

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

void backwardPropagation(const vector<float> &input,
                         const ForwardResult &forward_result,
                         const vector<float> &target,
                         Layer &inputToHidden,
                         Layer &hiddenToOutput)
{
    vector<float> output_delta(output_size);
    for (int o = 0; o < output_size; o++)
    {
        float error = forward_result.output[o] - target[o];
        output_delta[o] = error * sigmoid_derivative(forward_result.output_z[o]);
    }

    vector<float> hidden_delta(hidden_size);
    for (int h = 0; h < hidden_size; h++)
    {
        float error = 0.0f;
        for (int o = 0; o < output_size; o++)
        {
            error += output_delta[o] * hiddenToOutput.weights[h + o * hidden_size];
        }
        hidden_delta[h] = error * sigmoid_derivative(forward_result.hidden_z[h]);
    }

    for (int o = 0; o < output_size; o++)
    {
        for (int h = 0; h < hidden_size; h++)
        {
            float grad = output_delta[o] * forward_result.hidden[h];
            hiddenToOutput.weights[h + o * hidden_size] -= learning_rate * grad;
        }
        hiddenToOutput.biases[o] -= learning_rate * output_delta[o];
    }

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

vector<float> getTarget(int label)
{
    vector<float> target(output_size, 0.0f);
    target[label] = 1.0f;
    return target;
}

// 保存模型到文件
void saveModel(const Layer &inputToHidden, const Layer &hiddenToOutput, const string &filename)
{
    ofstream outFile(filename, ios::binary);
    if (!outFile)
    {
        cerr << "Error: Could not open file " << filename << " for writing." << endl;
        return;
    }

    // 保存输入到隐藏层的权重和偏置
    uint32_t weights_size = inputToHidden.weights.size();
    uint32_t biases_size = inputToHidden.biases.size();
    outFile.write(reinterpret_cast<const char *>(&weights_size), sizeof(weights_size));
    outFile.write(reinterpret_cast<const char *>(&biases_size), sizeof(biases_size));
    outFile.write(reinterpret_cast<const char *>(inputToHidden.weights.data()), weights_size * sizeof(float));
    outFile.write(reinterpret_cast<const char *>(inputToHidden.biases.data()), biases_size * sizeof(float));

    // 保存隐藏到输出层的权重和偏置
    weights_size = hiddenToOutput.weights.size();
    biases_size = hiddenToOutput.biases.size();
    outFile.write(reinterpret_cast<const char *>(&weights_size), sizeof(weights_size));
    outFile.write(reinterpret_cast<const char *>(&biases_size), sizeof(biases_size));
    outFile.write(reinterpret_cast<const char *>(hiddenToOutput.weights.data()), weights_size * sizeof(float));
    outFile.write(reinterpret_cast<const char *>(hiddenToOutput.biases.data()), biases_size * sizeof(float));

    outFile.close();
    cout << "Model saved to " << filename << endl;
}

int main()
{
    const int epochs = 300;
    vector<uint8_t> pixels;
    vector<float> pixelData;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    Layer inputToHidden;
    inputToHidden.weights.resize(input_size * hidden_size);
    for (auto &w : inputToHidden.weights)
    {
        w = dis(gen);
    }
    inputToHidden.biases.resize(hidden_size, 0.0f);

    Layer hiddenToOutput;
    hiddenToOutput.weights.resize(hidden_size * output_size);
    for (auto &w : hiddenToOutput.weights)
    {
        w = dis(gen);
    }
    hiddenToOutput.biases.resize(output_size, 0.0f);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
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

                vector<float> target = getTarget(i);
                ForwardResult forward_result = forwardPropagation(pixelData, inputToHidden, hiddenToOutput);
                backwardPropagation(pixelData, forward_result, target, inputToHidden, hiddenToOutput);
            }
        }
        cout << "Epoch " << epoch + 1 << " completed" << endl;
    }

    // 保存模型
    saveModel(inputToHidden, hiddenToOutput, "model.bin");

    return 0;
}