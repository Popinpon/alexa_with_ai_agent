import asyncio
from shared.smart_speaker_agent import create_smart_speaker_agent

async def test_chat_cycle():
    """会話サイクルをテスト"""
    agent = create_smart_speaker_agent()
    session_id = "test_session"
    
    # テスト1: 通常の質問
    print("=== テスト1: 通常の質問 ===")
    result1 = await agent.chat_cycle("川崎駅で遊べる場所を検索し、旅程をよく考えて、時間ごとにどこに行くか調べてください。また、明日の天気も調べてください。最後には17時までに川崎駅に着くようにしてください。", session_id)
    print(f"応答: {result1['prepared_response']}")
    print(f"継続必要: {result1['should_continue_processing']}")
    
    # テスト2: 継続要求のテスト
    if result1['should_continue_processing']:
        print("\n=== テスト2: 継続要求 ===")
        result2 = await agent.chat_cycle("はい", session_id)
        print(f"応答: {result2['prepared_response']}")
        print(f"継続必要: {result2['should_continue_processing']}")
    
    # テスト3: 新しい質問
    print("\n=== テスト3: 新しい質問 ===")
    result3 = await agent.chat_cycle("エアコンをつけて26度", session_id)
    print(f"応答: {result3['prepared_response']}")
    print(f"継続必要: {result3['should_continue_processing']}")

def main():
    asyncio.run(test_chat_cycle())

if __name__ == "__main__":
    main()