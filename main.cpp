#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <thread>
#include <chrono>   // 核心：用于计时
#include <iomanip>  // 核心：用于控制输出小数位数
#include "BertEmbedder.h" 

#ifdef _WIN32
#include <windows.h> // 用于设置 Windows 控制台编码
#endif

using namespace std;

// ==========================================
// 数据结构定义
struct KnowledgeItem {
    int id;             // 知识条目的唯一 ID
    string question;    //知识库中的问题文本
    string answer;      // 对应的答案文本
    vector<float> vec;  //预存问题文本经过模型计算后的句向量
};

// ==========================================
// 工具函数：从文件加载知识库
void load_knowledge_base(const string& filename, vector<KnowledgeItem>& kb) {
    ifstream file(filename);//打开数据文件
    if (!file.is_open()) {   //打开失败处理
        cerr << "Warning: 无法打开文件 " << filename << "，将仅使用内置默认数据。" << endl;
        return;
    }

    string line;
    int id_counter = 100; //外部文件的 ID 从 100 开始，防止和内置数据冲突
    while (getline(file, line)) { //逐行读取
        if (line.empty()) continue; // 跳过空行
        size_t delimiter_pos = line.find('|'); // 查找分隔符 '|'
        if (delimiter_pos != string::npos) {   // 如果找到了分隔符
            string q = line.substr(0, delimiter_pos); //分隔符前的是问题
            string a = line.substr(delimiter_pos + 1); // 分隔符后的是答案
            kb.push_back({ id_counter++, q, a, {} }); // 存入知识库数组中，向量暂时为空 `{}`
        }
    }
    cout << "成功从 " << filename << " 加载了 " << (id_counter - 100) << " 条数据。" << endl; // 【注释】报告加载数量
}

// ==========================================
// 工具函数：打字机效果输出
void type_writer_print(const string& text) {
    for (char c : text) { // 逐字遍历答案字符串
        cout << c;//输出单个字符
        this_thread::sleep_for(chrono::milliseconds(15)); //线程休眠 15 毫秒，制造打字机视觉延迟感
    }
    cout << endl; //输出完毕后换行
}

int main() {
    // 1. 设置控制台编码为 UTF-8
#ifdef _WIN32
    SetConsoleOutputCP(65001); // 设置 Windows 控制台输出编码为 UTF-8 (防止中文乱码)
    SetConsoleCP(65001);       // 设置 Windows 控制台输入编码为 UTF-8
#endif

    try {
        // 定义文件路径变量
        string model_path = "models_new/model.onnx";
        string vocab_path = "models_new/vocab.txt";
        string data_path = "data.txt";

        cout << "正在初始化 AI 引擎..." << endl;
        // 实例化嵌入层类，这会触发初始化模型和读取词典文件
        BertEmbedder embedder(model_path, vocab_path);

        // 2.构建知识库
        vector<KnowledgeItem> kb; // 创建存放知识条目的数组

        // 2.1核心保底数据
        // 手动压入几条硬编码的基础问答数据
        kb.push_back({ 1, "你好", "您好！我是小雅智能，很高兴为您服务。" });
        kb.push_back({ 2, "苹果手机", "Apple是全球知名的手机公司" });
        kb.push_back({ 3, "你是谁", "我是基于 BERT 模型构建的本地 C++ 混合检索系统，也叫小雅智能。" });
        kb.push_back({ 4, "你会做什么", "我可以进行语义分析、关键词匹配，为您提供精准的知识问答服务。" });
        kb.push_back({ 5, "C++难学吗", "C++ 确实有一定难度，特别是内存管理，但它是高性能系统的基石。" });
        kb.push_back({ 6, "再见", "再见！祝您答辩顺利！" });

        // 2.2从文件加载扩展数据
        load_knowledge_base(data_path, kb); // 调用工具函数加载外部 `data.txt` 文件的数据

        // 3.预计算阶段 (建立索引)
        cout << "正在构建混合索引 (计算向量 + 建立元数据)..." << endl;

        auto index_start = chrono::high_resolution_clock::now(); // 记录开始时间戳
        // 遍历所有知识库条目，使用模型计算每个问题的句向量并保存
        for (auto& item : kb) {
            item.vec = embedder.get_embedding(item.question);
        }
        auto index_end = chrono::high_resolution_clock::now(); // 记录结束时间戳
        auto index_duration = chrono::duration_cast<chrono::milliseconds>(index_end - index_start); // 【注释】计算差值(耗时)

        cout << "索引构建完成！共 " << kb.size() << " 条数据。" << endl;
        cout << "索引耗时: " << index_duration.count() << " ms\n" << endl; // 打印预计算耗时

        // 4. 进入问答循环
        cout << "================ 智能混合检索系统启动 ================" << endl;
        cout << "请输入您的问题 (输入 'q' 退出):" << endl;

        string user_input; // 保存用户在控制台的输入字符串

        //用于统计平均时间
        long long total_search_time_ms = 0; // 总耗时，初始化用于计算性能指标的变量
        int query_count = 0;                // 查询次数

        // [核心工具] UTF-8 字符拆分函数 (无论是 KMP 还是 LCS，处理中文都必须先拆字)
        // 使用 Lambda 表达式定义一个按字拆分字符串的函数
        auto split_utf8 = [](const string& text) -> vector<string> {
            vector<string> chars; // 存储拆分后的单个字符数组
            size_t i = 0;
            while (i < text.length()) { // 遍历输入字符串
                unsigned char c = text[i];
                int len = 1;
                // 识别当前 UTF-8 字符由几个字节组成（逻辑同 Tokenizer.cpp）
                if ((c & 0xE0) == 0xC0) len = 2;
                else if ((c & 0xF0) == 0xE0) len = 3;
                else if ((c & 0xF8) == 0xF0) len = 4;

                if (i + len > text.length()) len = text.length() - i; //防止越界
                string ch = text.substr(i, len); // 截取出这个完整的字符

                // 过滤掉常见的标点符号，防止标点符号干扰文本重合度的计算
                if (ch != "？" && ch != "！" && ch != "，" && ch != "。" &&
                    ch != "?" && ch != "!" && ch != "," && ch != ".") {
                    chars.push_back(ch);
                }
                i += len; // 索引步进
            }
            return chars; //返回字符切片数组
            };

        //系统主循环，持续等待用户输入
        while (true) {
            cout << "\n>> 用户: ";
            getline(cin, user_input); // 从控制台读取整行输入

            if (user_input == "q" || user_input == "Q") break; // 退出命令识别
            if (user_input.empty()) continue; // 如果是空回车则重试

            // ==========================================
            // [计时开始]
            // ==========================================
            auto search_start = chrono::high_resolution_clock::now(); // 记录检索开始时间

            // 4.1 计算查询向量
            vector<float> input_vec = embedder.get_embedding(user_input); // 将用户输入实时转换为句向量

            // 提前将用户输入按字拆分
            vector<string> user_chars = split_utf8(user_input); // 为关键字检索做准备：将用户输入拆分为单字

            // 4.2 混合检索
            float max_final_score = -1.0f; // 全局记录当前最高分数
            int best_match_index = -1;     //全局记录当前最佳匹配项在 kb 中的索引

            const float W_VECTOR = 0.7f;  //设定向量语义得分的权重为 70%
            const float W_KEYWORD = 0.3f; // 设定字面关键词得分的权重为 30%

            // 核心遍历：将用户的输入与知识库里的每一条数据进行打分比对
            for (size_t i = 0; i < kb.size(); ++i) {
                // A. 语义分数
                // 调用前面 Utils 中的余弦相似度，比较两个向量的距离作为语义分
                float vector_score = cosine_similarity(input_vec, kb[i].vec);

                // B. 关键词分数 (优化算法：最长公共子序列 LCS，类似 KMP 的顺序匹配但允许冗余)
                float keyword_score = 0.0f; // 初始化该知识项的关键词得分为0

                // 1. 捷径：如果完全一样或严格包含，直接满分
                if (kb[i].question.find(user_input) != string::npos ||
                    user_input.find(kb[i].question) != string::npos) {
                    keyword_score = 1.0f; // 触发字面短路机制，强制关键词满分
                }
                // 2. 否则，使用动态规划计算最长公共子序列 (LCS)
                else {
                    vector<string> kb_chars = split_utf8(kb[i].question); // 将被比较的知识库问题也拆分成单字
                    int m = kb_chars.size(); //知识库字数
                    int n = user_chars.size(); // 用户输入字数

                    if (m > 0 && n > 0) {
                        // 创建一个 (m+1) x (n+1) 的二维动态规划数组，初始值为 0
                        // 动态规划的 dp 表
                        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

                        // 填充 DP 矩阵
                        //嵌套循环比较每一个字
                        for (int r = 1; r <= m; ++r) {
                            for (int c = 1; c <= n; ++c) {
                                // 如果当前字符匹配成功，继承左上角的值加 1
                                if (kb_chars[r - 1] == user_chars[c - 1]) {
                                    dp[r][c] = dp[r - 1][c - 1] + 1; //状态转移方程：匹配
                                }
                                // 如果不匹配，取上方或左方的最大值 (允许跳过字符/模糊匹配)
                                else {
                                    dp[r][c] = max(dp[r - 1][c], dp[r][c - 1]); //状态转移方程：不匹配
                                }
                            }
                        }

                        // dp[m][n] 中存储的就是两个句子之间按顺序匹配的最大公共字数
                        int lcs_length = dp[m][n]; //获取最终的最长公共子序列长度

                        // 计算按顺序匹配的重合度比例
                        float ratio = static_cast<float>(lcs_length) / m; // 相对于知识库问题，覆盖了多少

                        // 如果按顺序包含的字符超过 50%，则赋予相应的关键词得分
                        if (ratio >= 0.5f) {
                            keyword_score = ratio; // 过滤掉低匹配，只有高覆盖率才起效
                        }
                    }
                }

                // C. 加权融合
                // 计算最终的混合得分
                float final_score = (vector_score * W_VECTOR) + (keyword_score * W_KEYWORD);

                //纠错机制：如果字面完全不匹配(0分)，但语义极其相似(>0.7)，退化为纯语义检索。
                // 这样可以解决类似 “你叫什么名字” 和 “你是谁” 这种字面毫无关系但意思一样的问题。
                if (keyword_score == 0.0f && vector_score > 0.7f) {
                    final_score = vector_score; // 纯语义回退
                }

                // 更新最高分和最佳匹配的索引值
                if (final_score >= max_final_score) {
                    max_final_score = final_score;
                    best_match_index = i;
                }
            }

            // ==========================================
            // [计时结束] & [性能指标计算]
            // ==========================================
            auto search_end = chrono::high_resolution_clock::now(); // 记录检索结束时间戳
            auto duration = chrono::duration_cast<chrono::milliseconds>(search_end - search_start); // 计算本次耗时

            // 累加数据
            long long current_time = duration.count(); //获取毫秒数
            total_search_time_ms += current_time;      // 累加到系统总耗时
            query_count++;                             //查询次数 +1

            // 计算平均值
            double avg_time = (double)total_search_time_ms / query_count; //得出平均延迟

            // 计算 QPS (Queries Per Second) - 理论每秒处理数
            // 防止除以0
            double qps = (current_time > 0) ? (1000.0 / current_time) : 999.0; //基于本次耗时估算并发吞吐量

            // 4.3 输出结果
            cout << "--------------------------------------------------" << endl;

            // [核心] 打印所有可能用到的性能指标
            cout << fixed << setprecision(2); // 设置小数显示两位，控制 cout 浮点数的格式
            cout << "【性能监控】" << endl;
            //打印各项性能参数
            cout << "  - 本次耗时: " << current_time << " ms" << endl;
            cout << "  - 平均耗时: " << avg_time << " ms (共 " << query_count << " 次)" << endl;
            cout << "  - 理论 QPS: " << qps << " (次/秒)" << endl;

            //判断最高分是否达到了置信度阈值 (0.6)，如果达到说明找到了答案
            if (best_match_index != -1 && max_final_score > 0.6) {
                cout << "【检索匹配】: " << kb[best_match_index].question << endl; //打印被匹配到的问题
                cout << "【置信度】: " << max_final_score << endl;//打印具体的得分
                cout << "【ych的AI回答】: ";
                type_writer_print(kb[best_match_index].answer);//调用打字机效果打印答案文本
                cout << "成功读取问答数据，且能正常接收并处理" << endl;

            }
            else {
                // 如果没有命中任何阈值超过 0.6 的答案，走兜底拒答逻辑
                cout << "【ych的AI回答】: 抱歉，知识库中未找到相关内容 (最高置信度: " << max_final_score << ")" << endl;
            }
            cout << "--------------------------------------------------" << endl;
        }

    }
    catch (const exception& e) { //捕获所有的标准异常（例如文件读取失败、模型崩溃等）
        cerr << "系统错误: " << e.what() << endl; //打印错误具体原因
    }

    return 0;
}