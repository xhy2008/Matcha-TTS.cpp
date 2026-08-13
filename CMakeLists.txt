cmake_minimum_required(VERSION 3.18)
project(matcha_zh_en_tts CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  # 源码与字符串字面量按 UTF-8 解析(含中文)
  add_compile_options(/utf-8)
endif()

# ONNXRuntime 运行时(仅此一个第三方依赖; espeak-ng 为动态加载, 不写进导入表)
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime")

# ---------------- 核心链接库: tts_engine.dll + tts_engine.lib ----------------
add_library(tts_engine SHARED src/tts_engine.cc)
target_include_directories(tts_engine
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${ORT_ROOT}/include")
target_link_directories(tts_engine PRIVATE "${ORT_ROOT}/lib")
target_link_libraries(tts_engine PRIVATE onnxruntime)
target_compile_definitions(tts_engine PRIVATE TTS_ENGINE_EXPORTS)

# 指定输出名与产物目录(便于直接引用)
set_target_properties(tts_engine PROPERTIES
  OUTPUT_NAME "tts_engine"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---------------- 命令行演示程序(链接 tts_engine.lib) ----------------
add_executable(tts_cli src/cli.cc)
target_include_directories(tts_cli PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_directories(tts_cli PRIVATE "${CMAKE_BINARY_DIR}/lib")
target_link_libraries(tts_cli PRIVATE tts_engine)
set_target_properties(tts_cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---------------- 运行时 DLL 拷贝到产物目录 ----------------
# onnxruntime.dll(tts_engine 的导入依赖)
file(GLOB ORT_RUNTIME_DLLS "${ORT_ROOT}/bin/*.dll")
# libespeak-ng.dll(前端 G2P, 运行时动态加载)
set(ESPEAK_DLL
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/libespeak-ng.dll")

foreach(tgt tts_engine tts_cli)
  add_custom_command(TARGET ${tgt} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_RUNTIME_DLLS}
            $<TARGET_FILE_DIR:${tgt}>
    COMMENT "Copying onnxruntime.dll")
  if(EXISTS "${ESPEAK_DLL}")
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ESPEAK_DLL}"
              $<TARGET_FILE_DIR:${tgt}>
      COMMENT "Copying libespeak-ng.dll")
  endif()
endforeach()
