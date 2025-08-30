import asyncio
from shared.smart_speaker_agent import create_smart_speaker_agent



#メイン関数
def main():
    agent = create_smart_speaker_agent()
    message =  asyncio.run(agent.chat("部屋の温度は？", "test_session"))
    print(f"Agent Response: {message}")
    # ここでエージェントを使用した処理を追加

# メイン処理
if __name__ == "__main__":
    main()