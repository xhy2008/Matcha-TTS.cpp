// tts_engine.cc — MatchaTTS zh-en 纯 C API 链接库实现
//
// 零框架依赖: 文本前端(规范化/分词/查音)自实现, 英文 G2P 动态加载 libespeak-ng.dll,
// 声学模型 + 声码器直接调用 ONNXRuntime(预优化模型 + ORT_DISABLE_ALL, 不重复优化),
// ISTFT/静音压缩/输出自实现。不依赖 sherpa-onnx 等任何上层框架。

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define _USE_MATH_DEFINES
#include <windows.h>

#include "tts_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

namespace {

// ============================================================================
// UTF-8 工具
// ============================================================================
// 纯逐字符切分(用于音素串, 对应 sherpa SplitTokensUTF8; 不做字母合并)
std::vector<std::string> SplitTokensUtf8(const std::string &s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    size_t len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
    out.push_back(s.substr(i, len));
    i += len;
  }
  return out;
}

// sherpa 的 SplitUtf8: 先按字符切分, 再把连续 ASCII 字母合并成英文词,
// 空格被丢弃, 标点/中文独立成词(与 matcha-tts-lexicon 的 MergeCharactersIntoWords 一致)
std::vector<std::string> SplitUtf8(const std::string &s) {
  std::vector<std::string> chars = SplitTokensUtf8(s);

  // 2 字节字符是否属于带附加符号的拉丁字母(可并入英文词, 对应 sherpa IsSpecial)
  auto IsSpecial2 = [](const std::string &w) {
    if (w.size() != 2) return false;
    unsigned char b0 = static_cast<unsigned char>(w[0]);
    unsigned char b1 = static_cast<unsigned char>(w[1]);
    char32_t cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F);
    return (cp >= 0xC0 && cp <= 0x24F) || (cp >= 0x1E00 && cp <= 0x1EFF) ||
           (cp == 0x2C6) || (cp == 0x2C7) || (cp == 0x2C9) || (cp == 0x2CA) ||
           (cp == 0x2CB) || (cp == 0x2D9) || (cp == 0x2DA) || (cp == 0x2DC) ||
           (cp == 0x2DD);
  };
  // 与 sherpa IsPunct(char) 一致: 撇号不算标点(可留在英文词内, 如 don't)
  auto IsPunctChar = [](char c) {
    return c != '\'' && std::ispunct(static_cast<unsigned char>(c));
  };

  std::vector<std::string> out;
  int32_t n = static_cast<int32_t>(chars.size());
  int32_t i = 0;
  int32_t prev = -1;
  while (i < n) {
    const std::string &w = chars[i];
    if (w.size() >= 3 || (w.size() == 2 && !IsSpecial2(w)) ||
        (w.size() == 1 &&
         (IsPunctChar(w[0]) || std::isspace(static_cast<unsigned char>(w[0]))))) {
      if (prev != -1) {
        std::string t;
        for (; prev < i; ++prev) t.append(chars[prev]);
        prev = -1;
        out.push_back(std::move(t));
      }
      if (!std::isspace(static_cast<unsigned char>(w[0]))) out.push_back(w);
      ++i;
      continue;
    }
    // 1 字节字母或 2 字节拉丁附加符号: 累积成词
    if (prev == -1) prev = i;
    ++i;
  }
  if (prev != -1) {
    std::string t;
    for (; prev < i; ++prev) t.append(chars[prev]);
    out.push_back(std::move(t));
  }
  return out;
}

static bool IsCJK(char32_t cp) {
  return (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x2E80 && cp <= 0xA4CF) ||
         (cp >= 0xA840 && cp <= 0xD7AF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
         (cp >= 0xFE30 && cp <= 0xFE4F) || (cp >= 0xFF65 && cp <= 0xFFDC) ||
         (cp >= 0x20000 && cp <= 0x2FFFF);
}

bool ContainsCJKStr(const std::string &s) {
  for (auto &c : SplitUtf8(s)) {
    char32_t cp = 0;
    const unsigned char *p = reinterpret_cast<const unsigned char *>(c.data());
    if (p[0] < 0x80) cp = p[0];
    else if ((p[0] & 0xE0) == 0xC0) cp = ((p[0] & 0x1F) << 6) | (p[1] & 0x3F);
    else if ((p[0] & 0xF0) == 0xE0)
      cp = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    else
      cp = ((p[0] & 0x07) << 18) | ((p[1] & 0x3F) << 12) |
           ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
    if (IsCJK(cp)) return true;
  }
  return false;
}

std::string ToLowerCase(std::string s) {
  for (auto &c : s) {
    if (c >= 'A' && c <= 'Z') c += 32;
  }
  return s;
}

bool IsPunctStr(const std::string &s) {
  static const std::unordered_set<std::string> puncts = {
      ",",  ".",  "!",  "?", ":", "\"", "'", "，",
      "。", "！", "？", "“", "”", "‘",  "’",
  };
  return puncts.count(s);
}

bool IsAlphaOrPunctByte(int ch) {
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
         (ch >= 0x21 && ch <= 0x2F) || (ch >= 0x3A && ch <= 0x40) ||
         (ch >= 0x5B && ch <= 0x60) || (ch >= 0x7B && ch <= 0x7E);
}

// ============================================================================
// 中文数字/日期/电话 规范化(近似替代 sherpa 的 3 个 FST)
// 规律(取自真实输出):
//   4位数字+年 → 逐位读(二零二五); 数字+月/号/日 → 数值读(十二月四号)
//   拨打/或者+数字 / 11位手机号 → 逐位读(幺幺零)
//   其余数字 → 中文数值(十二万三千四百五十六)
// ============================================================================
const std::vector<std::string> kDigits = {
    "零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};

std::string NumberToChinese(const std::string &digits) {
  size_t p = digits.find_first_not_of('0');
  if (p == std::string::npos) return "零";
  std::string d = digits.substr(p);
  static const std::vector<std::string> kD = {"零", "一", "二", "三", "四",
                                               "五", "六", "七", "八", "九"};
  static const std::vector<std::string> kU = {"", "十", "百", "千"};
  static const std::vector<std::string> kG = {"", "万", "亿", "万亿"};

  int len = static_cast<int>(d.size());
  int ngroups = (len + 3) / 4;
  int first_len = len % 4;
  if (first_len == 0) first_len = 4;

  std::string res;
  int pos = 0;
  for (int g = ngroups - 1; g >= 0; --g) {
    int glen = (g == ngroups - 1) ? first_len : 4;
    std::string gs = d.substr(pos, glen);
    pos += glen;

    // 低位组以 0 开头(如 10002 的 "0002") → 前面补零
    if (!res.empty() && gs[0] == '0') res += "零";

    std::string gstr;
    bool emitted = false;
    bool need_zero = false;
    for (int i = 0; i < glen; ++i) {
      int digit = gs[i] - '0';
      int mag = glen - 1 - i;  // 0=个 1=十 2=百 3=千
      if (digit == 0) {
        if (emitted && mag > 0) need_zero = true;
      } else {
        if (need_zero) {
          gstr += "零";
          need_zero = false;
        }
        // "一十"开头的 1 不读"一"(仅当十位且组内此前未输出)
        if (!(digit == 1 && mag == 1 && !emitted)) gstr += kD[digit];
        if (mag > 0) gstr += kU[mag];
        emitted = true;
      }
    }
    if (!emitted) {
      if (ngroups == 1) return "零";
      continue;  // 整组为零: 靠前面的补零逻辑衔接
    }
    res += gstr;
    if (g > 0) res += kG[g];
  }
  return res;
}

std::string DigitsToPhoneStyle(const std::string &digits) {
  // sherpa phone-zh FST: 前导连续 1 读"幺", 之后的 1 读"一"(如 189202512043 → 幺...一...)
  std::string res;
  bool seen_non_one = false;
  for (char c : digits) {
    if (c == '1') {
      res += seen_non_one ? "一" : "幺";
    } else {
      seen_non_one = true;
      res += kDigits[c - '0'];
    }
  }
  return res;
}

std::string DigitsToDigitStyle(const std::string &digits) {
  std::string res;
  for (char c : digits) res += kDigits[c - '0'];
  return res;
}

bool IsDigit(char c) { return c >= '0' && c <= '9'; }

std::string NormalizeText(const std::string &text) {
  std::string out;
  size_t i = 0;
  const size_t n = text.size();
  while (i < n) {
    if (!IsDigit(text[i])) {
      out += text[i];
      ++i;
      continue;
    }
    // 收集数字串
    size_t j = i;
    while (j < n && IsDigit(text[j])) ++j;
    std::string digits = text.substr(i, j - i);
    // 看数字后面的字
    std::string next;
    if (j < n) {
      unsigned char c = text[j];
      size_t len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
      next = text.substr(j, len);
    }
    // 前两个字符(用于"拨打/或者"判断)
    std::string prev2;
    if (i > 0) {
      size_t k = i;
      for (int cnt = 0; cnt < 2 && k > 0; ++cnt) {
        size_t k2 = k - 1;
        while (k2 > 0 && ((text[k2] & 0xC0) == 0x80)) --k2;
        prev2.insert(0, text.substr(k2, k - k2));
        k = k2;
      }
    }

    bool phone = false;
    bool digit_style = false;
    if (next == "年" && digits.size() == 4) {
      digit_style = true;  // 2025年 → 二零二五年
    } else if (prev2 == "拨打" || prev2 == "或者" ||
               (digits.size() == 11 && digits[0] == '1') || digits == "110" ||
               digits == "119" || digits == "120") {
      phone = true;
    }
    // 数字+月/号/日 走数值读法(默认分支)

    if (phone) {
      out += DigitsToPhoneStyle(digits);
    } else if (digit_style) {
      out += DigitsToDigitStyle(digits);
    } else {
      out += NumberToChinese(digits);
    }
    i = j;
  }
  return out;
}

// ============================================================================
// espeak-ng DLL 直调
// ============================================================================
class EspeakNg {
 public:
  ~EspeakNg() {
    if (terminate_ && dll_) terminate_();
    if (dll_) FreeLibrary(dll_);
  }

  bool Load(const std::string &dll_path, const std::string &data_dir) {
    dll_ = LoadLibraryA(dll_path.c_str());
    if (!dll_) return false;
    init_ = reinterpret_cast<int (*)(int, int, const char *, int)>(
        reinterpret_cast<void *>(GetProcAddress(dll_, "espeak_Initialize")));
    ttphonemes_ =
        reinterpret_cast<const char *(*)(const void **, const char *, int, int)>(
            reinterpret_cast<void *>(
                GetProcAddress(dll_, "espeak_TextToPhonemes")));
    set_voice_ = reinterpret_cast<int (*)(const char *)>(
        reinterpret_cast<void *>(GetProcAddress(dll_, "espeak_SetVoiceByName")));
    terminate_ = reinterpret_cast<int (*)(void)>(
        reinterpret_cast<void *>(GetProcAddress(dll_, "espeak_Terminate")));
    if (!init_ || !ttphonemes_ || !set_voice_) return false;

    int sample_rate = init_(0 /*AUDIO_OUTPUT_SYNCHRONOUS*/, 0,
                            data_dir.c_str(), 0);
    if (sample_rate != 22050) return false;
    if (set_voice_("en-us") != 0) return false;
    return true;
  }

  // 返回应用替换表后的音素符号列表(与 sherpa 的 ProcessPhonemes 一致)
  // 注意1: piper-phonemize(被 sherpa 编译进 DLL) 的 espeak 输出真 IPA,
  //        而 stock espeak-ng DLL 的 espeak_TextToPhonemes 只输出音素记忆码,
  //        因此先用 MnemonicToIpa 把记忆码还原为 raw IPA, 再做替换
  // 注意2: textptr 必须指向文本(由 espeak 推进), 传 nullptr 会退化为逐字母输出
  std::vector<std::string> PhonemizeSymbols(const std::string &word) {
    const void *ptr = word.c_str();
    std::string joined;
    while (ptr) {
      const char *phon =
          ttphonemes_(&ptr, word.c_str(), 0 /*AUTO*/, 0x02 /*IPA*/);
      if (!phon) break;
      joined += phon;
    }
    if (joined.empty()) return {};
    joined = MnemonicToIpa(joined);
    joined = ApplyReplacements(joined);
    return SplitTokensUtf8(joined);
  }

 private:
  // espeak-ng 音素记忆码 -> raw IPA(最长匹配; 已在 106 个英文词上与 sherpa 输出比对一致)
  static std::string MnemonicToIpa(const std::string &m) {
    static const std::pair<const char *, const char *> kMap[] = {
        // 3 字符
        {"i@3", "ɪr"}, {"aU@", "aʊr"}, {"aI@", "aɪr"},
        // 2 字符: 长元音/双元音/卷舌/弱化/辅音变体
        {"a:", "ɑː"}, {"A:", "ɑː"}, {"e:", "eː"}, {"i:", "iː"}, {"o:", "oː"},
        {"u:", "uː"}, {"3:", "ɜː"}, {"O:", "ɔː"}, {"0:", "ɑː"}, {"E:", "ɛː"},
        {"I:", "ɪː"}, {"U:", "ʊː"}, {"V:", "ʌː"},
        {"aU", "aʊ"}, {"oU", "oʊ"}, {"aI", "aɪ"}, {"eI", "eɪ"}, {"OI", "ɔɪ"},
        {"EI", "eɪ"}, {"oI", "ɔɪ"}, {"uI", "ʊɪ"}, {"oU#", "oʊ"}, {"aU#", "aʊ"},
        {"eI#", "eɪ"}, {"aI#", "aɪ"},
        {"A@", "ɑr"}, {"O@", "ɔr"}, {"e@", "ɛr"}, {"i@", "ɪr"}, {"U@", "ʊr"},
        {"o@", "or"}, {"3@", "ɜr"}, {"u@", "ur"}, {"aI3", "aɪr"}, {"aU3", "aʊr"},
        {"e@3", "ɛr"}, {"IR", "ɪr"}, {"VR", "ʌr"}, {"A~", "ɑ̃"}, {"O~", "ɔ̃"},
        {"@2", "ə"}, {"@5", "ə"}, {"@L", "əl"}, {"@#", "ə"}, {"02", "ɑ"},
        {"0#", "ɐ"}, {"a#", "ɐ"}, {"I2", "ɪ"}, {"I#", "ᵻ"}, {"I2#", "ɪ"},
        {"O2", "ɔ"}, {"E2", "ɛ"}, {"e#", "ɛ"}, {"a2", "ɑ"}, {"aa", "ɑ"},
        {"A#", "ɑ"}, {"tS#", "tʃ"}, {"dZ#", "dʒ"}, {"t#", "ɾ"}, {"d#", "d"},
        {"t2", "t"}, {"d2", "d"}, {"k2", "k"}, {"g2", "ɡ"}, {"p2", "p"},
        {"s2", "s"}, {"z2", "z"}, {"S#", "ʃ"}, {"Z#", "ʒ"}, {"N#", "ŋ"},
        {"n#", "n"}, {"l#", "l"}, {"r#", "ɹ"}, {"m#", "m"}, {"kh", "k"},
        {"ph", "p"}, {"th", "t"}, {"ts", "ts"}, {"dz", "dz"}, {"k#", "k"},
        {"g#", "ɡ"}, {"b#", "b"}, {"p#", "p"}, {"v#", "v"}, {"f#", "f"},
        {"s#", "s"}, {"z#", "z"}, {"w#", "w"}, {"j#", "j"}, {"h#", "h"},
        {"r-", "r"}, {"m@", "mə"}, {"n@", "nə"}, {"N@", "ŋə"}, {"l@", "lə"},
        {"r@", "rə"}, {"tS", "tʃ"}, {"dZ", "dʒ"},
        // 1 字符
        {"ː", ""}, {"'", "ˈ"}, {",", "ˌ"},
        {"a", "æ"}, {"e", "e"}, {"E", "ɛ"}, {"I", "ɪ"}, {"O", "ɔ"}, {"U", "ʊ"},
        {"V", "ʌ"}, {"A", "ɑ"}, {"i", "i"}, {"o", "o"}, {"u", "u"}, {"y", "y"},
        {"Y", "y"}, {"1", "ə"}, {"2", "ɵ"}, {"8", "ɜ"}, {"@", "ə"}, {"0", "ɑ"},
        {"3", "ɚ"}, {"L", "l"}, {"C", "ç"}, {"Q", "q"}, {"J", "ɟ"}, {"G", "ɣ"},
        {"R", "ʁ"}, {"x", "x"}, {"X", "χ"},
        {"p", "p"}, {"b", "b"}, {"t", "t"}, {"d", "d"}, {"k", "k"}, {"g", "g"},
        {"f", "f"}, {"v", "v"}, {"T", "θ"}, {"D", "ð"}, {"s", "s"}, {"z", "z"},
        {"S", "ʃ"}, {"Z", "ʒ"}, {"h", "h"}, {"m", "m"}, {"n", "n"}, {"N", "ŋ"},
        {"l", "l"}, {"r", "r"}, {"j", "j"}, {"w", "w"},
    };
    std::string out;
    size_t i = 0;
    while (i < m.size()) {
      bool hit = false;
      for (int len = 3; len >= 1; --len) {
        if (i + len > m.size()) continue;
        std::string sub = m.substr(i, len);
        for (const auto &p : kMap) {
          if (std::strlen(p.first) == static_cast<size_t>(len) &&
              sub == p.first) {
            out += p.second;
            i += len;
            hit = true;
            break;
          }
        }
        if (hit) break;
      }
      if (!hit) {  // 未知字符原样保留, 避免死循环
        out += m[i];
        i += 1;
      }
    }
    return out;
  }

  static std::string ApplyReplacements(std::string s) {
    static const std::vector<std::pair<std::string, std::string>> reps = {
        {"ɝ", "ɜɹ"}, {"ɚ", "əɹ"}, {"eɪ", "A"}, {"aɪ", "I"},
        {"ɔɪ", "Y"}, {"oʊ", "O"}, {"əʊ", "O"}, {"aʊ", "W"},
        {"tʃ", "ʧ"}, {"dʒ", "ʤ"}, {"ː", ""},   {"g", "ɡ"},
        {"r", "ɹ"},  {"e", "ɛ"},
    };
    for (const auto &p : reps) {
      size_t pos = 0;
      while ((pos = s.find(p.first, pos)) != std::string::npos) {
        s.replace(pos, p.first.size(), p.second);
        pos += p.second.size();
      }
    }
    return s;
  }

  HMODULE dll_ = nullptr;
  int (*init_)(int, int, const char *, int) = nullptr;
  const char *(*ttphonemes_)(const void **, const char *, int, int) = nullptr;
  int (*set_voice_)(const char *) = nullptr;
  int (*terminate_)(void) = nullptr;
};

// ============================================================================
// 分词器(MatchaTtsLexicon 复刻)
// ============================================================================
class Tokenizer {
 public:
  bool Load(const std::string &tokens_path, const std::string &lexicon_path,
            const std::string &espeak_dll, const std::string &espeak_data) {
    {
      std::ifstream is(tokens_path);
      std::string line;
      while (std::getline(is, line)) {
        std::istringstream iss(line);
        std::string sym;
        int id = -1;
        iss >> sym;
        if (iss.eof()) {
          id = std::atoi(sym.c_str());
          sym = " ";
        } else {
          iss >> id;
        }
        token2id_[sym] = id;
      }
    }
    {
      std::ifstream is(lexicon_path);
      std::string line;
      while (std::getline(is, line)) {
        std::istringstream iss(line);
        std::string word;
        std::vector<std::string> phones;
        iss >> word;
        if (word.empty()) continue;
        word = ToLowerCase(word);
        if (word2ids_.count(word)) continue;
        std::string ph;
        std::vector<int32_t> ids;
        bool ok = true;
        while (iss >> ph) {
          auto it = token2id_.find(ph);
          if (it == token2id_.end()) {
            ok = false;
            break;
          }
          ids.push_back(it->second);
        }
        if (!ok || ids.empty()) continue;
        word2ids_[word] = ids;
        all_words_.insert(word);
      }
    }
    blank_id_ = token2id_[" "];
    return espeak_.Load(espeak_dll, espeak_data);
  }

  // 与 sherpa ConvertTextToTokenIds 等价; 返回句子序列
  std::vector<std::vector<int64_t>> ConvertTextToTokenIds(
      const std::string &_text) {
    std::string text = _text;
    // 标点替换 + 空格合并
    static const std::vector<std::pair<std::string, std::string>> punct_rep = {
        {"，", ","}, {"、", ","}, {"；", ";"}, {"：", ","},   {":", ","},
        {"。", "."}, {"？", "?"}, {"！", "!"},
    };
    for (const auto &p : punct_rep) {
      size_t pos = 0;
      while ((pos = text.find(p.first, pos)) != std::string::npos) {
        text.replace(pos, p.first.size(), p.second);
        pos += p.second.size();
      }
    }
    // 合并连续空白
    {
      std::string t;
      bool last_space = false;
      for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
          if (!last_space) t += ' ';
          last_space = true;
        } else {
          t += c;
          last_space = false;
        }
      }
      text = t;
    }

    std::vector<std::string> words = SplitUtf8(text);
    // 去掉标点/空白后的多余空格
    std::vector<std::string> words2 = std::move(words);
    words.clear();
    for (size_t i = 0; i < words2.size(); ++i) {
      if (i == 0) {
        words.push_back(std::move(words2[i]));
      } else if (words2[i] == " ") {
        if (words.back() == " " || IsPunctStr(words.back())) continue;
        words.push_back(std::move(words2[i]));
      } else if (IsPunctStr(words2[i])) {
        if (words.back() == " " || IsPunctStr(words.back())) continue;
        words.push_back(std::move(words2[i]));
      } else {
        words.push_back(std::move(words2[i]));
      }
    }

    // PhraseMatcher: 最长词典匹配(首字符非字母/标点时)
    std::vector<std::string> phrases;
    {
      const int kMaxSearchLen = 10;
      int32_t num = static_cast<int32_t>(words.size());
      for (int32_t i = 0; i < num;) {
        int32_t start = i;
        std::string w;
        if (!IsAlphaOrPunctByte(static_cast<unsigned char>(words[i][0]))) {
          int32_t end = std::min(i + kMaxSearchLen - 1, num - 1);
          while (end > start) {
            std::string cand;
            for (int32_t k = start; k <= end; ++k) cand += words[k];
            if (IsAlphaOrPunctByte(
                    static_cast<unsigned char>(cand.back()))) {
              --end;
              continue;
            }
            if (all_words_.count(cand)) {
              i = end + 1;
              w = cand;
              break;
            }
            --end;
          }
        }
        if (w.empty()) {
          w = words[i];
          i += 1;
        }
        phrases.push_back(std::move(w));
      }
    }

    std::vector<std::vector<int64_t>> ans;
    std::vector<int64_t> this_sentence;
    for (const std::string &w : phrases) {
      std::vector<int32_t> ids = ConvertWordToIds(w);
      if (ids.empty()) {
        last_word_ = w;
        continue;
      }
      if (!last_word_.empty() && IsAlphaByte(last_word_[0])) {
        this_sentence.push_back(blank_id_);
      }
      for (int32_t id : ids) this_sentence.push_back(id);
      if (IsPunctStr(w)) {
        ans.push_back(std::move(this_sentence));
        this_sentence = {};
      }
      last_word_ = w;
    }
    if (!this_sentence.empty()) ans.push_back(std::move(this_sentence));
    last_word_ = "";
    return ans;
  }

 private:
  static bool IsAlphaByte(int c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  }

  std::vector<int32_t> ConvertWordToIds(const std::string &w) {
    std::vector<int32_t> ans;
    auto it = word2ids_.find(w);
    if (it != word2ids_.end()) return it->second;
    auto it2 = token2id_.find(w);
    if (it2 != token2id_.end()) return {it2->second};
    if (ContainsCJKStr(w)) {
      for (const auto &ch : SplitUtf8(w)) {
        if (word2ids_.count(ch)) {
          auto sub = ConvertWordToIds(ch);
          ans.insert(ans.end(), sub.begin(), sub.end());
        }
      }
    } else {
      // 英文词: espeak-ng
      auto syms = espeak_.PhonemizeSymbols(w);
      for (const auto &p : syms) {
        auto it3 = token2id_.find(p);
        if (it3 != token2id_.end()) ans.push_back(it3->second);
      }
    }
    return ans;
  }

  std::unordered_map<std::string, int32_t> token2id_;
  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;
  std::unordered_set<std::string> all_words_;
  EspeakNg espeak_;
  int32_t blank_id_ = 1;
  std::string last_word_;
};

// ============================================================================
// ONNX Runtime 推理
// ============================================================================
std::vector<float> RunAcoustic(Ort::Session &sess, const std::vector<int64_t> &x,
                               int64_t &mel_frames) {
  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> in;
  std::vector<int64_t> x_shape{1, static_cast<int64_t>(x.size())};
  in.push_back(Ort::Value::CreateTensor<int64_t>(
      mem, const_cast<int64_t *>(x.data()), x.size(), x_shape.data(),
      x_shape.size()));
  int64_t len = static_cast<int64_t>(x.size());
  int64_t len_shape[1] = {1};
  in.push_back(Ort::Value::CreateTensor<int64_t>(mem, &len, 1, len_shape, 1));
  float noise_scale = 0.667f, length_scale = 1.0f;
  int64_t s1[1] = {1};
  in.push_back(Ort::Value::CreateTensor<float>(mem, &noise_scale, 1, s1, 1));
  in.push_back(Ort::Value::CreateTensor<float>(mem, &length_scale, 1, s1, 1));
  const char *in_names[] = {"x", "x_length", "noise_scale", "length_scale"};
  const char *out_names[] = {"mel"};
  std::vector<Ort::Value> out(1);
  sess.Run(Ort::RunOptions{nullptr}, in_names, in.data(), in.size(), out_names,
           out.data(), 1);
  auto info = out[0].GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = info.GetShape();
  mel_frames = shape[2];
  size_t n = info.GetElementCount();
  const float *p = out[0].GetTensorData<float>();
  return std::vector<float>(p, p + n);
}

// 返回 mag/x/y (各自 [1, bins, frames])
void RunVocoder(Ort::Session &sess, const std::vector<float> &mel,
                int64_t mel_frames, std::vector<float> *mag,
                std::vector<float> *cx, std::vector<float> *cy) {
  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> in;
  std::vector<int64_t> mshape{1, 80, mel_frames};
  in.push_back(Ort::Value::CreateTensor<float>(
      mem, const_cast<float *>(mel.data()), mel.size(), mshape.data(), 3));
  const char *in_names[] = {"mels"};
  const char *out_names[] = {"mag", "x", "y"};
  std::vector<Ort::Value> out(3);
  sess.Run(Ort::RunOptions{nullptr}, in_names, in.data(), in.size(), out_names,
           out.data(), 3);
  auto copy = [&](Ort::Value &v, std::vector<float> *dst) {
    auto info = v.GetTensorTypeAndShapeInfo();
    size_t n = info.GetElementCount();
    const float *p = v.GetTensorData<float>();
    dst->assign(p, p + n);
  };
  copy(out[0], mag);
  copy(out[1], cx);
  copy(out[2], cy);
}

// ============================================================================
// ISTFT(kaldi-native-fbank 复刻): n_fft=1024, hop=256, hann(periodic), center
// ============================================================================
void Radix2Ifft(std::vector<std::complex<double>> &a) {
  int n = static_cast<int>(a.size());
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (int len = 2; len <= n; len <<= 1) {
    double ang = 2.0 * M_PI / len;  // 逆变换取正号
    std::complex<double> wlen(std::cos(ang), std::sin(ang));
    for (int i = 0; i < n; i += len) {
      std::complex<double> w(1, 0);
      for (int k = 0; k < len / 2; ++k) {
        std::complex<double> u = a[i + k];
        std::complex<double> v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
}

std::vector<float> IStft(const std::vector<float> &real,
                         const std::vector<float> &imag, int n_fft, int hop,
                         int num_frames) {
  // 预计算 hann 窗(periodic)
  std::vector<float> win(n_fft);
  for (int i = 0; i < n_fft; ++i) {
    win[i] = 0.5f - 0.5f * std::cos(2.0 * M_PI * i / n_fft);
  }
  int num_samples = n_fft + (num_frames - 1) * hop;
  std::vector<float> samples(num_samples, 0.0f);
  std::vector<float> denom(num_samples, 0.0f);
  int bins = n_fft / 2 + 1;

  for (int f = 0; f < num_frames; ++f) {
    std::vector<std::complex<double>> buf(n_fft);
    const float *pr = real.data() + f * bins;
    const float *pi = imag.data() + f * bins;
    // 与 knf 相同的打包: DC 和 Nyquist 都在实部
    buf[0] = std::complex<double>(pr[0], 0);
    for (int i = 1; i < n_fft / 2; ++i) {
      buf[i] = std::complex<double>(pr[i], pi[i]);
    }
    buf[n_fft / 2] = std::complex<double>(pr[n_fft / 2], 0);
    for (int i = n_fft / 2 + 1; i < n_fft; ++i) {
      buf[i] = std::conj(buf[n_fft - i]);
    }
    Radix2Ifft(buf);
    // 1/n 缩放 + 加窗 + 叠接相加
    float *ps = samples.data() + f * hop;
    float *pd = denom.data() + f * hop;
    for (int i = 0; i < n_fft; ++i) {
      float x = static_cast<float>(buf[i].real()) / n_fft * win[i];
      ps[i] += x;
      pd[i] += win[i] * win[i];
    }
  }

  for (int i = 0; i < num_samples; ++i) {
    if (denom[i] != 0) samples[i] /= denom[i];
  }
  // center: 裁掉两侧 n_fft/2
  return std::vector<float>(samples.begin() + n_fft / 2,
                            samples.end() - n_fft / 2);
}

// ============================================================================
// ScaleSilence(sherpa 复刻, silence_scale=0.2)
// ============================================================================
std::vector<float> ScaleSilence(const std::vector<float> &samples,
                                int sample_rate, float scale) {
  if (scale == 1.0f) return samples;
  int threshold = static_cast<int>(sample_rate * 0.2);
  std::vector<std::pair<int, int>> intervals;
  int last = -1;
  int n = static_cast<int>(samples.size());
  for (int i = 0; i < n; ++i) {
    if (std::fabs(samples[i]) <= 0.01f) {
      if (last == -1) last = i;
      continue;
    }
    if (last != -1 && i - last < threshold) {
      last = -1;
      continue;
    }
    if (last != -1) {
      intervals.push_back({last, i});
      last = -1;
    }
  }
  if (last != -1 && n - last > threshold) intervals.push_back({last, n});
  if (intervals.empty()) return samples;
  std::vector<float> ans;
  ans.reserve(samples.size());
  int i = 0;
  for (const auto &iv : intervals) {
    ans.insert(ans.end(), samples.begin() + i, samples.begin() + iv.first);
    i = iv.second;
    int k = static_cast<int>((iv.second - iv.first) * scale);
    ans.insert(ans.end(), samples.begin() + iv.first,
               samples.begin() + iv.first + k);
  }
  if (i < n) ans.insert(ans.end(), samples.begin() + i, samples.end());
  return ans;
}

}  // namespace

// ============================================================================
// 实例(前端 + 两个 ONNX Session)
// ============================================================================
struct MatchaTts {
  Tokenizer tokenizer;
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::Session> acoustic;
  std::unique_ptr<Ort::Session> vocoder;

  int sample_rate = 16000;
  int n_fft = 1024;
  int hop = 256;
};

// ============================================================================
// C API
// ============================================================================
void matcha_tts_config_default(MatchaTtsConfig *cfg) {
  std::memset(cfg, 0, sizeof(*cfg));
}

int matcha_tts_create(MatchaTts **out, const MatchaTtsConfig *cfg, char *err,
                      int err_size) {
  if (err && err_size > 0) err[0] = '\0';
  auto fail = [&](const std::string &msg) {
    if (err && err_size > 0) {
      std::snprintf(err, err_size, "%s", msg.c_str());
    }
    return -1;
  };

  MatchaTtsConfig d;
  matcha_tts_config_default(&d);
  if (!cfg) cfg = &d;

  const char *tokens = cfg->tokens_path ? cfg->tokens_path
                                        : "models/matcha-icefall-zh-en/tokens.txt";
  const char *lexicon = cfg->lexicon_path ? cfg->lexicon_path
                                          : "models/matcha-icefall-zh-en/lexicon.txt";
  const char *espeak_dll = cfg->espeak_dll ? cfg->espeak_dll : "libespeak-ng.dll";
  const char *espeak_data =
      cfg->espeak_data ? cfg->espeak_data
                       : "models/matcha-icefall-zh-en/espeak-ng-data";
  const char *acoustic_model =
      cfg->acoustic_model
          ? cfg->acoustic_model
          : "models/matcha-icefall-zh-en/model-steps-3.opt.onnx";
  const char *vocoder_model = cfg->vocoder_model
                                  ? cfg->vocoder_model
                                  : "models/vocos-16khz-univ.opt.onnx";
  int threads = cfg->intra_op_threads > 0 ? cfg->intra_op_threads : 1;

  std::unique_ptr<MatchaTts> tts(new MatchaTts);

  if (!tts->tokenizer.Load(tokens, lexicon, espeak_dll, espeak_data)) {
    return fail("failed to load text frontend (tokens/lexicon/espeak-ng)");
  }

  try {
    tts->env.reset(new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "tts_engine"));
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(threads);
    so.SetInterOpNumThreads(1);
    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    // 预优化模型 + 禁用图优化 → 启动不做重复优化
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    auto ToWide = [](const char *s) {
      int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
      std::wstring w(n - 1, L'\0');
      MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
      return w;
    };
    tts->acoustic.reset(new Ort::Session(*tts->env, ToWide(acoustic_model).c_str(), so));
    tts->vocoder.reset(new Ort::Session(*tts->env, ToWide(vocoder_model).c_str(), so));
  } catch (const Ort::Exception &e) {
    return fail(std::string("failed to load ONNX model: ") + e.what());
  }

  if (cfg->single_core) {
    DWORD_PTR mask = 1ULL << (cfg->cpu_index >= 0 ? cfg->cpu_index : 0);
    SetProcessAffinityMask(GetCurrentProcess(), mask);
  }

  *out = tts.release();
  return 0;
}

int matcha_tts_tokenize(MatchaTts *tts, const char *text, int64_t *tokens,
                        int max_tokens, int *sentence_splits) {
  if (!tts || !text) return -1;
  std::string norm = NormalizeText(text);
  auto seqs = tts->tokenizer.ConvertTextToTokenIds(norm);
  int pos = 0;
  for (size_t s = 0; s < seqs.size(); ++s) {
    if (sentence_splits) sentence_splits[s] = static_cast<int>(seqs[s].size());
    if (tokens && max_tokens > 0) {
      for (size_t k = 0; k < seqs[s].size(); ++k) {
        if (pos >= max_tokens) return -2;  // 缓冲不足
        tokens[pos++] = seqs[s][k];
      }
      if (s + 1 < seqs.size() && pos < max_tokens) tokens[pos++] = 0;
    }
  }
  return static_cast<int>(seqs.size());
}

int matcha_tts_synthesize(MatchaTts *tts, const char *text, float **samples,
                          int *num_samples, int *sample_rate) {
  if (!tts || !text) return -1;
  if (samples) *samples = nullptr;
  if (num_samples) *num_samples = 0;
  if (sample_rate) *sample_rate = tts->sample_rate;

  std::string norm = NormalizeText(text);
  auto seqs = tts->tokenizer.ConvertTextToTokenIds(norm);

  std::vector<float> audio;
  for (auto &x : seqs) {
    int64_t mf = 0;
    std::vector<float> mel = RunAcoustic(*tts->acoustic, x, mf);
    std::vector<float> mag, cx, cy;
    RunVocoder(*tts->vocoder, mel, mf, &mag, &cx, &cy);
    // mag/cx/cy: [bins, frames], 转成 real/imag 行优先 [frames, bins]
    int bins = static_cast<int>(cy.size()) / static_cast<int>(mf);
    std::vector<float> real(static_cast<size_t>(mf) * bins);
    std::vector<float> imag(static_cast<size_t>(mf) * bins);
    for (int f = 0; f < static_cast<int>(mf); ++f) {
      for (int b = 0; b < bins; ++b) {
        float m = mag[b * mf + f];
        real[f * bins + b] = m * cx[b * mf + f];
        imag[f * bins + b] = m * cy[b * mf + f];
      }
    }
    auto wav = IStft(real, imag, tts->n_fft, tts->hop, static_cast<int>(mf));
    audio.insert(audio.end(), wav.begin(), wav.end());
  }
  audio = ScaleSilence(audio, tts->sample_rate, 0.2f);

  if (samples && !audio.empty()) {
    float *buf = static_cast<float *>(std::malloc(audio.size() * sizeof(float)));
    if (!buf) return -3;
    std::memcpy(buf, audio.data(), audio.size() * sizeof(float));
    *samples = buf;
  }
  if (num_samples) *num_samples = static_cast<int>(audio.size());
  return 0;
}

void matcha_tts_free_samples(float *samples) { std::free(samples); }

void matcha_tts_destroy(MatchaTts *tts) { delete tts; }
