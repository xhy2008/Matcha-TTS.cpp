// cli.cc — tts_engine 链接库的命令行演示程序
//
// 用法:
//   tts_cli.exe --text "文本" [--outdir out] [--single-core] [--cpu 0]
//   tts_cli.exe --file texts.txt [--repeat 3]
//   tts_cli.exe --print-tokens --text "我最近在学习machine learning。"
//
// 编译后运行目录需与 tts_engine.dll、onnxruntime.dll、libespeak-ng.dll
// 及默认模型路径(models/...) 相对摆放。

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "tts_engine.h"

static double NowMs() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char *argv[]) {
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);

  std::string text_arg, file_arg, outdir = "output";
  int repeat = 1;
  bool single_core = false, print_tokens = false;
  int cpu_index = 0;

  std::vector<std::string> args(argv + 1, argv + argc);
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &a = args[i];
    if (a == "--text" && i + 1 < args.size()) text_arg = args[++i];
    else if (a == "--file" && i + 1 < args.size()) file_arg = args[++i];
    else if (a == "--outdir" && i + 1 < args.size()) outdir = args[++i];
    else if (a == "--repeat" && i + 1 < args.size())
      repeat = std::atoi(args[++i].c_str());
    else if (a == "--single-core") single_core = true;
    else if (a == "--print-tokens") print_tokens = true;
    else if (a == "--cpu" && i + 1 < args.size())
      cpu_index = std::atoi(args[++i].c_str());
    else {
      std::cerr << "未知参数: " << a << "\n";
      return 1;
    }
  }
  if (repeat < 1) repeat = 1;

  // ---------- 创建引擎 ----------
  char err[512];
  MatchaTtsConfig cfg;
  matcha_tts_config_default(&cfg);
  cfg.single_core = single_core ? 1 : 0;
  cfg.cpu_index = cpu_index;

  MatchaTts *tts = nullptr;
  double t0 = NowMs();
  if (matcha_tts_create(&tts, &cfg, err, sizeof(err)) != 0) {
    std::cerr << "创建引擎失败: " << err << "\n";
    return 1;
  }
  std::cout << "引擎加载: " << (NowMs() - t0) << " ms\n";

  // ---------- 文本来源 ----------
  std::vector<std::string> sentences;
  if (!text_arg.empty()) {
    sentences.push_back(text_arg);
  } else if (!file_arg.empty()) {
    std::ifstream fin(file_arg);
    std::string line;
    while (std::getline(fin, line)) if (!line.empty()) sentences.push_back(line);
  } else {
    sentences = {
        "我最近在学习machine learning，希望能够在未来的artificial "
        "intelligence领域有所建树。",
        "在这次vocation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
        "开始数字测试。2025年12月4号，拨打110或者189202512043。123456块钱。",
        "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; "
        "经济不断增长。",
        "今天天气真好，Hello World，欢迎来到AI时代。The quick brown fox "
        "jumps over the lazy dog. 这是一个中英文混合测试。",
    };
  }

  // ---------- 分词/合成 ----------
  CreateDirectoryA(outdir.c_str(), nullptr);
  for (size_t s = 0; s < sentences.size(); ++s) {
    if (print_tokens) {
      std::cout << "SEN " << s << ": '" << sentences[s] << "'\n";
      std::vector<int> splits(128);
      int n = matcha_tts_tokenize(tts, sentences[s].c_str(), nullptr, 0,
                                  splits.data());
      std::vector<int64_t> all(8192);
      matcha_tts_tokenize(tts, sentences[s].c_str(), all.data(),
                          static_cast<int>(all.size()), nullptr);
      int pos = 0;
      for (int k = 0; k < n && k < 128; ++k) {
        std::cout << "  sub" << k << ":";
        for (int j = 0; j < splits[k] && pos < static_cast<int>(all.size());
             ++j)
          std::cout << " " << all[pos++];
        if (k + 1 < n) ++pos;  // 跳过子句分隔 0
        std::cout << "\n";
      }
      continue;
    }

    for (int r = 0; r < repeat; ++r) {
      float *samples = nullptr;
      int num = 0, sr = 0;
      double t1 = NowMs();
      int rc = matcha_tts_synthesize(tts, sentences[s].c_str(), &samples,
                                     &num, &sr);
      double ms = NowMs() - t1;
      if (rc != 0) {
        std::cerr << "句子 " << (s + 1) << " 合成失败: " << rc << "\n";
        continue;
      }
      std::printf("句子 %zu: 合成 %.0f ms | 音频 %.3f s | RTF = %.3f\n",
                  s + 1, ms, static_cast<double>(num) / sr,
                  ms / 1000.0 / (static_cast<double>(num) / sr));

      char wav_name[256];
      std::snprintf(wav_name, sizeof(wav_name), "%s\\sentence_%02zu.wav",
                    outdir.c_str(), s + 1);
      // 写 PCM16 WAV
      std::ofstream f(wav_name, std::ios::binary);
      uint32_t data_size = static_cast<uint32_t>(num) * 2;
      uint32_t total = 36 + data_size;
      auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<char *>(&v), 4); };
      auto w16 = [&](uint16_t v) { f.write(reinterpret_cast<char *>(&v), 2); };
      f.write("RIFF", 4);
      w32(total);
      f.write("WAVE", 4);
      f.write("fmt ", 4);
      w32(16);
      w16(1);
      w16(1);
      w32(static_cast<uint32_t>(sr));
      w32(static_cast<uint32_t>(sr) * 2);
      w16(2);
      w16(16);
      f.write("data", 4);
      w32(data_size);
      for (int i = 0; i < num; ++i) {
        float v = std::max(-1.0f, std::min(1.0f, samples[i]));
        int16_t pcm = static_cast<int16_t>(v * 32767);
        f.write(reinterpret_cast<char *>(&pcm), 2);
      }
      matcha_tts_free_samples(samples);
      std::cout << "  已保存: " << wav_name << "\n";
    }
  }

  matcha_tts_destroy(tts);
  return 0;
}
