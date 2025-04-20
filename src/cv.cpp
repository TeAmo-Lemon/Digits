#include <iostream>
#include <fstream>
#include <sstream>
#include <windows.h> // 用于设置控制台编码
using namespace std;

int main()
{
    // 设置控制台输出为 UTF-8 编码
    // SetConsoleOutputCP(CP_UTF8);

    // 逐行读取csv文件
    ifstream file("../public/mnist_train.csv");
    if (!file.is_open())
    {
        cerr << "Unable to open file" << endl;
        return 1;
    }

    string line;
    while (getline(file, line))
    {
        // 输出读取到的每一行
        cout << sizeof(line) << endl;
    }

    cout << "hello" << endl;

    return 0;
}
