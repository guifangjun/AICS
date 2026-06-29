"""
命令行测试脚本：验证 LLM 连通性和基础对话。

运行方式：
  python scripts/test_chat.py

前置条件：
  已设置对应 Provider 的环境变量（如 DEEPSEEK_API_KEY）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from core.llm_factory import create_chat_model


def test_llm():
    print(f"Provider: {config.llm.provider}")
    print(f"Model: {config.llm.model_name}")
    print(f"Base URL: {config.llm.base_url}")
    print(f"Temperature: {config.llm.temperature}")
    print("-" * 40)

    print("正在创建 LLM 实例...")
    try:
        llm = create_chat_model()
        print("LLM 实例创建成功！")
    except ValueError as e:
        print(f"[错误] 请检查环境变量: {e}")
        return

    print("正在测试对话...")
    test_messages = [
        "你好，请问你是谁？",
        "帮我用一句话介绍下人工智能",
    ]

    for msg in test_messages:
        print(f"\n[用户]: {msg}")
        try:
            response = llm.invoke(msg)
            print(f"[助手]: {response.content}")
        except Exception as e:
            print(f"[错误] API 调用失败: {e}")
            return

    print("\n" + "=" * 40)
    print("测试完成！LLM 连通正常。")
    print("如需切换模型，修改 config/config.yaml 中的 llm.provider 即可。")


if __name__ == "__main__":
    test_llm()
