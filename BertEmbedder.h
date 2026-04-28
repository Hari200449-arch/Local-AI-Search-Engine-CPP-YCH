#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <onnxruntime_cxx_api.h> // 必须包含 ONNX Runtime 头文件
#include <fstream>  // 新增：用于读取文件
#include <algorithm>
#include <thread>   // 新增：用于打字机延时
#include <chrono>   // 新增：用于时间控制
using namespace std;

// 类定义
class BertEmbedder {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    map<string, int> vocab;
    int hidden_size = 768;

public:
    // 构造函数
    BertEmbedder(const string& model_path, const string& vocab_path);

    // 核心功能
    void load_vocab(const string& vocab_path);
    vector<int64_t> tokenize(const string& text);
    vector<float> get_embedding(const string& text);

    // 辅助函数 (在 Tokenizer.cpp 实现，但 Init.cpp 也要用，所以设为 public 或 private 均可，只要类内可见)
    bool SafeIsSpace(char c);
    bool SafeIsAlnum(char c);
    char SafeToLower(char c);
};

// ==========================================
// 全局工具函数声明 (对应 Utils.cpp)
// ==========================================
// 这一步非常重要！如果不声明，main.cpp 就会报“找不到 identifier”
float cosine_similarity(const vector<float>& A, const vector<float>& B);
vector<string> split(const string& s, char delimiter);
