#include "BertEmbedder.h"

//获取输入文本的句向量（Sentence Embedding）
vector<float> BertEmbedder::get_embedding(const string& text) {
    vector<int64_t> input_ids = tokenize(text); //调用分词器，将文本转为 Token ID 数组
    size_t seq_len = input_ids.size();          //获取序列长度

    //BERT 模型需要的三个核心输入之二和三：
    vector<int64_t> attention_mask(seq_len, 1); // 注意力掩码，全为 1 表示所有 Token 都需要被关注（这里没有做 padding 处理，所以都是有效词）
    vector<int64_t> token_type_ids(seq_len, 0); //句子类型 ID，单句输入时全为 0

    //定义 ONNX 模型的输入输出节点名称（必须和导出模型时的名称严格一致）
    vector<const char*> input_names = { "input_ids", "attention_mask", "token_type_ids" };
    vector<const char*> output_names = { "sentence_embedding" }; //预期输出的节点名
    vector<int64_t> input_shape = { 1, (int64_t)seq_len };       //输入张量的形状：[batch_size=1, sequence_length]

    //创建一个 CPU 内存分配器信息，告诉 ONNX 运行时在哪里分配内存
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    //创建准备送入模型的输入张量集合
    vector<Ort::Value> input_tensors;
    //将 input_ids 转换为 ONNX 的 Tensor 格式并加入集合
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()));
    //将 attention_mask 转换为 ONNX 的 Tensor 格式并加入集合
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size()));
    //将 token_type_ids 转换为 ONNX 的 Tensor 格式并加入集合
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, token_type_ids.data(), token_type_ids.size(), input_shape.data(), input_shape.size()));

    //执行模型推理！传入输入节点名、输入张量、输入数量(3)，以及输出节点名、预期输出数量(1)
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
        input_names.data(), input_tensors.data(), 3,
        output_names.data(), 1);

    //从输出的 Tensor 中提取原始的 float 数据指针
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    //将指针对应的数据复制到 C++ 标准的 vector<float> 中（长度为隐藏层维度 hidden_size）
    vector<float> result(floatarr, floatarr + hidden_size);
    return result; //返回生成的句向量
}