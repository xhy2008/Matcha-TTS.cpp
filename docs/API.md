# MatchaTTS zh-en — 纯 C API 文档

零框架依赖的中英混读 TTS 链接库。文本前端（中文数字规范化、分词、词典查音、英文 G2P）
全部自实现；声学模型 + 声码器直接调用 ONNXRuntime；ISTFT、静音压缩、PCM 输出自实现。
**不依赖 sherpa-onnx 等任何上层框架**，与 sherpa-onnx 输出的 token ids 在测试语料上逐位一致。

## 目录结构

```
include/tts_engine.h   公共头文件(C API)
src/tts_engine.cc      链接库实现(前端 + ORT 推理 + ISTFT)
src/cli.cc             命令行演示程序(链接 tts_engine.lib)
deps/onnxruntime/      ONNXRuntime 1.23 头文件 / onnxruntime.lib / onnxruntime.dll
runtime/libespeak-ng.dll  英文 G2P 运行库(运行时动态加载)
models/
  matcha-icefall-zh-en/
    model-steps-3.opt.onnx   声学模型(已预优化)
    tokens.txt / lexicon.txt 分词资源
    espeak-ng-data/          espeak-ng 数据
  vocos-16khz-univ.opt.onnx  声码器模型(已预优化)
```

## 依赖说明

链接库 **tts_engine.dll** 的导入表仅依赖 `onnxruntime.dll`；
`libespeak-ng.dll` 在运行时通过 `LoadLibrary` 动态加载（G2P 用）。
除此之外无任何第三方运行时/框架依赖。全部路径（模型、词典、espeak DLL/数据）通过配置
传入，可自由部署。

## 构建

```sh
cmake -S . -B build -A x64
cmake --build build --config Release
```

产物：

| 文件 | 说明 |
|---|---|
| `build/bin/Release/tts_engine.dll` | 链接库本体 |
| `build/lib/Release/tts_engine.lib` | 导入库（链接用） |
| `build/bin/Release/tts_cli.exe` | 命令行演示程序 |
| `build/bin/Release/onnxruntime.dll` | 自动拷贝 |
| `build/bin/Release/libespeak-ng.dll` | 自动拷贝 |

> 运行时需保证模型路径 `models/...`（或通过配置指定）相对于工作目录可达。

## 链接库接口（C API）

头文件：`tts_engine.h`。所有函数为 `extern "C"`，导出名见下表。
错误码约定：`0` 成功；`< 0` 失败。

### 配置结构

```c
typedef struct MatchaTtsConfig {
  const char *tokens_path;    // tokens.txt,              默认 "models/matcha-icefall-zh-en/tokens.txt"
  const char *lexicon_path;   // lexicon.txt,             默认 "models/matcha-icefall-zh-en/lexicon.txt"
  const char *espeak_dll;     // libespeak-ng.dll 路径,    默认 "libespeak-ng.dll"
  const char *espeak_data;    // espeak-ng-data 目录,     默认 "models/matcha-icefall-zh-en/espeak-ng-data"
  const char *acoustic_model; // 声学模型(.opt.onnx),     默认 "models/matcha-icefall-zh-en/model-steps-3.opt.onnx"
  const char *vocoder_model;  // 声码器模型(.opt.onnx),   默认 "models/vocos-16khz-univ.opt.onnx"
  int single_core;            // 1 = 绑定单核(可选)
  int cpu_index;              // 绑核编号, 默认 0
  int intra_op_threads;       // ORT 推理线程数, 默认 1
} MatchaTtsConfig;
```

使用前先 `matcha_tts_config_default(&cfg)` 清零，再覆盖需要的字段；
`NULL` 字段在创建时取上述默认值。

### 函数列表

#### `void matcha_tts_config_default(MatchaTtsConfig *cfg)`

用零值填充配置（各字段取默认路径/参数）。

#### `int matcha_tts_create(MatchaTts **out, const MatchaTtsConfig *cfg, char *err, int err_size)`

创建引擎实例（加载分词资源 + 两个 ONNX Session）。耗时约 2~3 s（主要为 ONNX 加载）。

- `out` 成功时输出实例句柄，用 `matcha_tts_destroy` 释放
- `cfg` 可为 `NULL`（全部默认）
- `err/err_size` 失败时返回错误信息（可传 `NULL,0`）
- 返回 `0` 成功，`-1` 失败

加载时使用 `ORT_DISABLE_ALL` 图优化级别加载**预优化**模型，避免每次启动重复优化。

#### `int matcha_tts_tokenize(MatchaTts *tts, const char *text, int64_t *tokens, int max_tokens, int *sentence_splits)`

对文本分词，返回按标点切分的子句 token ids（与 sherpa-onnx 的 `ConvertTextToTokenIds`
等价）。数字（年/月/日/电话/金额）自动规范化读法。

- `tokens` 输出缓冲；子句之间用 `0` 分隔；可为 `NULL` 仅查询子句数
- `max_tokens` 缓冲容量（token 数）
- `sentence_splits` 可选，输出每个子句的 token 个数（数组长度 = 子句数）
- 返回子句数；缓冲不足返回 `-2`

#### `int matcha_tts_synthesize(MatchaTts *tts, const char *text, float **samples, int *num_samples, int *sample_rate)`

合成语音。

- `samples` 输出 PCM 采样（float, 1 通道），内部 `malloc`，用
  `matcha_tts_free_samples` 释放；不需要时传 `NULL`
- `num_samples` 输出采样数
- `sample_rate` 固定输出 `16000`
- 返回 `0` 成功，`-1` 参数错误，`-3` 内存不足

注意：实例非线程安全，同一实例的并发调用需要外部加锁（前端状态与 ORT Session 均为
单线程设计）。

#### `void matcha_tts_free_samples(float *samples)`

释放 `matcha_tts_synthesize` 返回的采样缓冲。

#### `void matcha_tts_destroy(MatchaTts *tts)`

销毁实例，释放全部资源。

## 使用示例

```c
#include "tts_engine.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  MatchaTtsConfig cfg;
  matcha_tts_config_default(&cfg);
  // cfg.single_core = 1;          // 可选
  // cfg.acoustic_model = "...";   // 自定义路径

  char err[256];
  MatchaTts *tts = NULL;
  if (matcha_tts_create(&tts, &cfg, err, sizeof(err)) != 0) {
    fprintf(stderr, "create failed: %s\n", err);
    return 1;
  }

  float *samples = NULL;
  int n = 0, sr = 0;
  if (matcha_tts_synthesize(tts,
      "你好，今天天气不错。Welcome to the AI era!",
      &samples, &n, &sr) != 0) {
    fprintf(stderr, "synthesize failed\n");
  } else {
    printf("got %d samples @ %d Hz\n", n, sr);
    // 将 samples[0..n) 写出为 WAV/PCM...
    matcha_tts_free_samples(samples);
  }

  matcha_tts_destroy(tts);
  return 0;
}
```

编译（链接 tts_engine.lib）：

```sh
cl /Iinclude demo.c build/lib/Release/tts_engine.lib
# 运行目录需有 tts_engine.dll / onnxruntime.dll / libespeak-ng.dll 及模型
```

## 命令行演示（tts_cli.exe）

```
tts_cli.exe --text "文本" [--outdir out] [--single-core] [--cpu 0]
tts_cli.exe --file texts.txt [--repeat N]
tts_cli.exe --print-tokens [--text "文本"]   # 打印 token ids(调试/校验)
```

无参数时合成内置 5 句中英混读测试语料。

## 已知限制

- 英文 G2P 的弱读/词典行为与 sherpa 使用的 piper-espeak 存在个别差异（如孤立词 "a"），
  常规句子无影响；数字规范化覆盖常见年份/日期/电话/金额场景
- 声学模型含随机噪声，同一文本两次合成的音频长度/波形不完全相同（属模型固有随机性）
