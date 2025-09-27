#!/usr/bin/env python3
"""
Smart Speaker Agent のテスト用エントリーポイント
"""
import asyncio
from shared import SmartSpeakerAgent

async def main():
    """テスト用メイン関数"""
    try:
        print("🚀 Smart Speaker Agent Test Starting...")
        agent = SmartSpeakerAgent("azure_openai")
        
        # テストクエリ
        queries = [
            "エアコン消して",
            "今日の天気は？"
        ]
        
        session_id = "test_session"
        
        for query in queries:
            print(f"\n=== 質問: {query} ===")
            response = await agent.chat(query, session_id)
            print(f"回答: {response}")
            
    except Exception as e:
        print(f"❌ テスト実行エラー: {str(e)}")
        print("💡 環境変数が設定されているか確認してください")

if __name__ == "__main__":
    asyncio.run(main())