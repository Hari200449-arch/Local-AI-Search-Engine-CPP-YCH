#include "BertEmbedder.h"

// 安全辅助函数实现
bool BertEmbedder::SafeIsSpace(char c) {
    //安全地判断字符是否为空格，强制转换为 unsigned char 防止越界引发断言错误
    return isspace(static_cast<unsigned char>(c));
}

bool BertEmbedder::SafeIsAlnum(char c) {
    //安全地判断字符是否为字母或数字
    return isalnum(static_cast<unsigned char>(c));
}

char BertEmbedder::SafeToLower(char c) {
    //安全地将大写字母转换为小写字母（BERT词表通常是小写的）
    return static_cast<char>(tolower(static_cast<unsigned char>(c)));
}

// 核心分词逻辑
vector<int64_t> BertEmbedder::tokenize(const string& text) {
    vector<int64_t> tokens; // 用于存储最终的 Token ID 序列
    tokens.push_back(101);  // BERT 模型的特殊起始符 [CLS] 的固定 ID 是 101

    size_t n = text.length(); //获取输入文本的总字节数
    size_t i = 0;             //初始化遍历索引

    while (i < n) { //遍历整个文本
        char c = text[i]; //获取当前字节
        if (SafeIsSpace(c)) { i++; continue; } //如果是空格，直接跳过处理下一个

        //判断是否是 ASCII 字符（英文字母、数字、基础标点等，ASCII 码小于 128 (0x80)）
        if ((unsigned char)c < 0x80) {
            // 处理英文和数字
            if (SafeIsAlnum(c)) {
                string buffer; //用于拼接连续的字母或数字
                // 循环提取连续的英数字符
                while (i < n && (unsigned char)text[i] < 0x80 && SafeIsAlnum(text[i])) {
                    buffer += SafeToLower(text[i]); //转为小写并拼接到 buffer
                    i++; //索引后移
                }
                // 如果这个单词在词表中存在，加入其对应的 ID
                if (vocab.count(buffer)) tokens.push_back(vocab[buffer]);
                else {
                    //如果不在词表中（OOV），则将其拆分成单字符进行查找（简化的 WordPiece 替代方案）
                    for (char x : buffer) {
                        string single(1, x);
                        if (vocab.count(single)) tokens.push_back(vocab[single]);
                        else tokens.push_back(100); //如果单字符也不在词表，用 [UNK] 未知字符的 ID 100 替代
                    }
                }
            }
            else {
                // 处理标点符号
                string single(1, c); //将单个标点转为字符串
                i++; //索引后移
                if (vocab.count(single)) tokens.push_back(vocab[single]); //查表
                else tokens.push_back(100); //找不到就给 [UNK] ID (100)
            }
        }
        else {
            // 处理中文信息(UTF-8编码解析)
            int len = 1; //记录当前中文字符占用的字节数
            unsigned char uc = (unsigned char)c;
            //根据 UTF-8 编码规则，通过首字节的二进制特征判断这个字符由几个字节组成
            if ((uc & 0xE0) == 0xC0) len = 2;       //110xxxxx，占2个字节
            else if ((uc & 0xF0) == 0xE0) len = 3;  //1110xxxx，占3个字节（大部分中文）
            else if ((uc & 0xF8) == 0xF0) len = 4;  //11110xxx，占4个字节（罕见字/Emoji）
            if (i + len > n) len = n - i; //防止截断导致的越界

            string char_str = text.substr(i, len); //根据计算出的长度，截取完整的 UTF-8 字符
            i += len; //索引跳过整个字符的字节长度
            if (vocab.count(char_str)) tokens.push_back(vocab[char_str]); //查词表并存入 ID
            else tokens.push_back(100); //找不到就给 [UNK] ID (100)
        }
    }
    tokens.push_back(102); //BERT 模型的特殊结束符 [SEP] 的固定 ID 是 102
    return tokens; //返回完整的分词 ID 数组
}