#include "BertEmbedder.h"
#include <fstream>
#include <iostream>

// 构造函数：初始化 ONNX Runtime 环境
BertEmbedder::BertEmbedder(const string& model_path, const string& vocab_path)
//初始化 ONNX 环境(env)，设置日志级别为 WARNING 以减少刷屏；初始化 session 指针为 null
    : env(ORT_LOGGING_LEVEL_WARNING, "BertEmbedder"), session(nullptr) {

    Ort::SessionOptions session_options; //创建会话配置对象
    session_options.SetIntraOpNumThreads(1); //限制单个算子的多线程数为 1（降低 CPU 瞬时占用率）
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC); //开启基础计算图优化

    //因为 Windows 系统对中文路径支持不好，所以要特殊处理一下文件路径的格式
#ifdef _WIN32
    size_t newSize = strlen(model_path.c_str()) + 1; // 计算需要转换的宽字符长度
    wchar_t* w_model_path = new wchar_t[newSize];    // 动态分配宽字符数组
    size_t convertedChars = 0;                       // 接收转换字符数的变量
    // 将多字节字符串(模型路径)转换为宽字符串(Windows API 常用的 wchar_t 格式)
    mbstowcs_s(&convertedChars, w_model_path, newSize, model_path.c_str(), _TRUNCATE);
    session = Ort::Session(env, w_model_path, session_options); // 使用宽字符路径加载 ONNX 模型文件
    delete[] w_model_path; // 释放动态分配的内存，防止内存泄漏
#else
    // 如果是 Linux/Mac 等类 Unix 系统，直接使用常规字符串路径加载
    session = Ort::Session(env, model_path.c_str(), session_options);
#endif
    cout << "模型成功加载，无报错" << endl; // 控制台提示

    //加载文件，把vocab.txt 里的字一个一个读进来
    load_vocab(vocab_path); // 调用词表加载函数
}

// 加载词表
void BertEmbedder::load_vocab(const string& vocab_path) {
    ifstream in(vocab_path); // 打开词表文件
    string line;             // 用于逐行存放读取的内容
    int id = 0;              // Token 的 ID 初始化为 0
    if (!in.is_open()) {     //检查文件是否成功打开
        cerr << "错误: 无法打开词表文件 " << vocab_path << endl; //报错提示
        return;
    }
    while (getline(in, line)) { //按行读取文件，直到结束
        if (!line.empty() && line.back() == '\r') line.pop_back(); //处理 Windows 换行符（CRLF），剔除末尾的 '\r'

        // 这里需要调用 SafeIsSpace，因为它在 Tokenizer 模块定义，但属于类成员，可以直接调用
        //去除行尾多余的不可见空白字符，防止词汇匹配失败
        while (!line.empty() && isspace((unsigned char)line.back())) {
            line.pop_back();
        }
        //如果处理后该行不为空，则将该词存入 Hash Map (vocab)，键为词本身，值为其对应的自增 ID
        if (!line.empty()) vocab[line] = id;
        id++; //ID 加 1，对应文件中的下一行
    }
    cout << "vocab.txt词表加载完成，共 " << vocab.size() << " 个词。" << endl; //控制台提示
}