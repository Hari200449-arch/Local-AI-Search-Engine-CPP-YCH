#include "BertEmbedder.h"
#include <cmath>
#include <sstream>

// 余弦相似度算法
float cosine_similarity(const vector<float>& A, const vector<float>& B) {
    float dot_product = 0.0f; //初始化向量的点积（分子部分）为 0
    float norm_a = 0.0f;//初始化向量 A 的模长平方为 0
    float norm_b = 0.0f;//初始化向量 B 的模长平方为 0

    //遍历向量的每一个维度（假设 A 和 B 维度相同）
    for (size_t i = 0; i < A.size(); ++i) {
        dot_product += A[i] * B[i]; //累加对应维度的乘积，计算点积
        norm_a += A[i] * A[i];// 累加 A 在各维度的平方
        norm_b += B[i] * B[i];//累加 B 在各维度的平方
    }

    //防止除零错误，如果任意一个向量全为0，则相似度为0
    if (norm_a == 0 || norm_b == 0) return 0.0f;
    // 返回余弦相似度：点积 / (A的模长 * B的模长)
    return dot_product / (sqrt(norm_a) * sqrt(norm_b));
}


// 字符串分割工具
vector<string> split(const string& s, char delimiter) {
    vector<string> tokens;// 用于存放分割后的子字符串数组
    string token;//用于临时存放当前正在提取的子字符串
    istringstream tokenStream(s); //将输入的字符串 s 转换为输入流，方便按字符读取
    //使用 getline 按指定的分隔符 delimiter 从流中读取一段段字符串
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token); //将读取到的子字符串存入结果数组
    }
    return tokens; //返回分割后的结果
}