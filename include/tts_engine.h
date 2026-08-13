// tts_engine.h — MatchaTTS zh-en 纯 C API
//
// 零框架依赖的中英混读 TTS 链接库:
//   - 文本前端(规范化/分词/查音) 自实现, 英文 G2P 动态加载 libespeak-ng.dll
//   - 声学模型 + 声码器直接调用 ONNXRuntime(预优化模型, 无重复图优化)
//   - ISTFT/静音压缩/PCM 自实现
//
// 链接库本身体现为 tts_engine.dll + tts_engine.lib:
//   使用者只需 include 本头文件并链接 tts_engine.lib;
//   运行目录需存在 onnxruntime.dll 与 libespeak-ng.dll 及模型/词典数据。

#ifndef TTS_ENGINE_H_
#define TTS_ENGINE_H_

#include <cstdint>

#ifdef _WIN32
#ifdef TTS_ENGINE_EXPORTS
#define TTS_API __declspec(dllexport)
#else
#define TTS_API __declspec(dllimport)
#endif
#define TTS_CALL __cdecl
#else
#define TTS_API
#define TTS_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 不透明句柄: 一个实例 = 一次前端 + 两个 ONNX Session(声学/声码器)
typedef struct MatchaTts MatchaTts;

// 创建参数(全部有默认值, 见 matcha_tts_config_default)
typedef struct MatchaTtsConfig {
  // 文本前端
  const char *tokens_path;    // tokens.txt           默认 "models/matcha-icefall-zh-en/tokens.txt"
  const char *lexicon_path;   // lexicon.txt          默认 "models/matcha-icefall-zh-en/lexicon.txt"
  const char *espeak_dll;     // libespeak-ng.dll 路径(动态加载) 默认 "libespeak-ng.dll"
  const char *espeak_data;    // espeak-ng-data 目录  默认 "models/matcha-icefall-zh-en/espeak-ng-data"
  // 推理模型(需为预优化 .opt.onnx; 加载时禁用图优化避免重复优化)
  const char *acoustic_model; // 声学模型             默认 "models/matcha-icefall-zh-en/model-steps-3.opt.onnx"
  const char *vocoder_model;  // 声码器模型           默认 "models/vocos-16khz-univ.opt.onnx"
  // 可选: 绑定单核(多线程无法进一步提升 RTF, 且结果更稳定)
  int single_core;            // 1 = 绑定到 cpu_index
  int cpu_index;              // 默认 0
  // 可选: 推理线程数(默认 1)
  int intra_op_threads;
} MatchaTtsConfig;

// 用默认值填充配置(NULL 字段在 create 时取默认)
TTS_API void TTS_CALL matcha_tts_config_default(MatchaTtsConfig *cfg);

// 创建实例。成功返回 0; 失败返回非 0 并把错误信息写入 err(最多 err_size 字节)。
TTS_API int TTS_CALL matcha_tts_create(MatchaTts **out,
                                       const MatchaTtsConfig *cfg, char *err,
                                       int err_size);

// 分词: 把 text 切成若干子句(按标点)并输出 token ids。
//   tokens         输出缓冲(子句间用 0 分隔), 可为 NULL 只查询
//   max_tokens     缓冲容量
//   sentence_splits 可选: 每个子句的 token 数(长度 = 子句数), 可为 NULL
//   返回子句数(失败返回负值)
TTS_API int TTS_CALL matcha_tts_tokenize(MatchaTts *tts, const char *text,
                                         int64_t *tokens, int max_tokens,
                                         int *sentence_splits);

// 合成: 中英混读文本 -> 16kHz float PCM。
//   samples   输出指针(内部 malloc, 用 matcha_tts_free_samples 释放), 可为 NULL
//   num_samples 输出采样数(1 通道, 16 kHz)
//   sample_rate 固定 16000
// 成功返回 0。
TTS_API int TTS_CALL matcha_tts_synthesize(MatchaTts *tts, const char *text,
                                           float **samples, int *num_samples,
                                           int *sample_rate);

// 释放 matcha_tts_synthesize 返回的 samples
TTS_API void TTS_CALL matcha_tts_free_samples(float *samples);

// 销毁实例
TTS_API void TTS_CALL matcha_tts_destroy(MatchaTts *tts);

#ifdef __cplusplus
}
#endif

#endif  // TTS_ENGINE_H_
