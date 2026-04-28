# Local-AI-Search-Engine-CPP-YCH
基于ONNX Runtime与C++的本地AI搜索引擎的设计与实现 - 毕业设计代码

💻 项目环境与使用说明
本项目最大的亮点在于彻底脱离了庞大的 Python 深度学习生态。系统的所有逻辑均使用纯 C++ 原生开发，具有极高的运行效率和极低的资源占用。

1. 如果你只想直接运行（面向普通用户）
无需安装任何复杂的代码环境，只需确保满足以下基本条件即可：

操作系统：Windows 10 或 Windows 11。

文件准备：请确保下载了编译好的 .exe 程序，并且把它和以下文件放在同一个文件夹内：

onnxruntime.dll（ONNX推理引擎的动态链接库文件）

models_new 文件夹（里面必须包含 model.onnx 模型文件和 vocab.txt 词表文件）

data.txt（知识库问答数据）

如何运行：双击 .exe 文件，看到控制台弹出“正在初始化 AI 引擎...”即可开始问答。

2. 如果你想编译源码（面向开发者）
如果你想修改我的代码并重新生成程序，你需要配置 C++ 的编译环境：

🛠️ 环境依赖要求
IDE/编译器：强烈推荐使用 Visual Studio 2022（包含 C++ 桌面开发工作负载）。

核心依赖库：ONNX Runtime C++ API（推荐下载 v1.14 或以上版本的 Release 包）。

⚙️ 编译配置步骤
下载依赖库：前往微软 ONNX Runtime 官方 GitHub 发布页，下载对应 Windows 的 C++ 压缩包，解压到你的电脑里。

在 VS 中配置头文件路径：

右键你的项目 -> 属性 -> C/C++ -> 常规 -> 附加包含目录。

把 ONNX Runtime 文件夹里的 include 目录路径添加进去（这就是为了让代码能认识 #include <onnxruntime_cxx_api.h>）。

在 VS 中配置链接库（Lib）：

属性 -> 链接器 -> 常规 -> 附加库目录，把 ONNX Runtime 文件夹里的 lib 目录添加进去。

属性 -> 链接器 -> 输入 -> 附加依赖项，手动输入 onnxruntime.lib。

编译并运行：

在 VS 顶部将模式切换为 Release / x64，然后点击“生成解决方案”。

⚠️ 关键注意：编译成功后，VS 会生成一个 .exe 文件。你必须手动把你下载的 ONNX Runtime 文件夹里的 onnxruntime.dll 文件，复制到这个 .exe 旁边，否则程序会闪退报错。
