# Matcha-TTS.cpp

基于 **ONNXRuntime** 的 MatchaTTS 社区 zh-en 模型的纯 C++ 中英混读 TTS 链接库。

不依赖 sherpa-onnx 等上层框架：文本前端（中文数字/日期/电话规范化、分词、词典查音、
英文 G2P）全部自实现，声学模型 + 声码器直接调用 ONNXRuntime，ISTFT 自实现。
分词结果与 sherpa-onnx 在测试语料上逐位一致。

## 特性

- **中英混读**：中文走词典（拼音）+ 英文走 espeak-ng G2P，同一句话内自由混排
- **零框架依赖**：链接库仅依赖 `onnxruntime.dll`（推理）+ `libespeak-ng.dll`（G2P，
  运行时动态加载），无 sherpa-onnx / 其他框架
- **无重复优化**：加载预优化 ONNX（`.opt.onnx`）并禁用运行时图优化，启动快
- **单核 CPU 实时**：RTF ≈ 0.16（16 kHz，AMD Ryzen 5 3500U 实测）
- **C API 链接库**：`tts_engine.dll` + `tts_engine.lib`，可被 C/C++/其他语言调用

## 目录结构

```
include/tts_engine.h   公共头文件(C API)
src/tts_engine.cc      链接库实现
src/cli.cc             命令行演示程序
docs/API.md            API 文档
deps/onnxruntime/      ONNXRuntime 1.23(头文件/lib/dll)
runtime/               libespeak-ng.dll(G2P 运行库)
models/                声学/声码器模型 + tokens/lexicon + espeak-ng-data
```

## 构建

```sh
cmake -S . -B build -A x64
cmake --build build --config Release
```

产物（`build/bin/Release/`）：`tts_engine.dll`、`tts_engine.lib`、`tts_cli.exe`，
并自动拷贝 `onnxruntime.dll` 与 `libespeak-ng.dll`。

## 快速使用

```sh
# 命令行合成(默认内置 5 句中英混读测试语料)
./build/bin/Release/tts_cli.exe --single-core

# 合成指定文本
./build/bin/Release/tts_cli.exe --text "你好，今天天气不错。Welcome to AI era!"

# 打印 token ids(调试/与 sherpa 校验)
./build/bin/Release/tts_cli.exe --print-tokens --text "我最近在学习machine learning。"
```

C 语言调用链接库：

```c
#include "tts_engine.h"

MatchaTtsConfig cfg;
matcha_tts_config_default(&cfg);
MatchaTts *tts = NULL;
if (matcha_tts_create(&tts, &cfg, err, sizeof(err)) != 0) { /* ... */ }

float *samples = NULL;
int n = 0, sr = 0;
matcha_tts_synthesize(tts, "Hello, world! 你好，世界！", &samples, &n, &sr);
/* samples: 16 kHz 单声道 float PCM, 用 matcha_tts_free_samples 释放 */
matcha_tts_destroy(tts);
```

完整 API 说明见 [docs/API.md](docs/API.md)。

## 模型准备

模型与词典不随仓库提交（体积大），需自行准备到 `models/` 下：

- 声学模型 `models/matcha-icefall-zh-en/model-steps-3.opt.onnx`
- 声码器模型 `models/vocos-16khz-univ.opt.onnx`
- 分词资源 `models/matcha-icefall-zh-en/tokens.txt`、`lexicon.txt`
- G2P 数据 `models/matcha-icefall-zh-en/espeak-ng-data/`

可从 sherpa-onnx 官方模型集（k2-fsa/sherpa-onnx 的 `matcha-icefall-zh-en` /
`vocos-16khz-univ`）下载原始 ONNX，再用 `ORT_DISABLE_ALL` 之外的一次性离线优化
（`SetOptimizedModelFilePath`）生成 `.opt.onnx`。

## 许可

本项目代码与文档采用 MIT 许可。注意：模型与 espeak-ng 各自有其原始许可，
请按其来源许可证使用。
