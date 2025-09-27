"""
Smart Speaker Agent - リファクタリング後のメインクラス
"""
import logging
import os
import time
from typing import Dict, List, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from shared.types import LLMProvider, AgentState, ConversationState
from shared.switchbot_manager import SwitchBotManager
from shared.conversation_manager import ConversationManager
from shared.workflow_builder import WorkflowBuilder
from shared.gemini_agent import GeminiAgent

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class SmartSpeakerAgent:
    """スマートスピーカーエージェントのメインクラス（リファクタリング後）"""
    
    def __init__(self, llm_provider: str = "azure_openai"):
        init_start = time.time()
        logger.info(f"🚀 SmartSpeakerAgent initialization started")
        
        self.llm_provider = llm_provider
        
        # LLM作成時間計測
        llm_start = time.time()
        self.llm = self._create_llm()
        llm_time = time.time() - llm_start
        logger.info(f"⏱️ LLM creation: {llm_time:.3f}s")
        
        # GeminiAgent作成時間計測
        gemini_start = time.time()
        self.gemini_search_agent = GeminiAgent()
        gemini_time = time.time() - gemini_start
        logger.info(f"⏱️ GeminiAgent creation: {gemini_time:.3f}s")
        
        # マネージャークラスの初期化
        self.switchbot_manager = SwitchBotManager()
        self.conversation_manager = ConversationManager(self.llm)
        
        # ツールとワークフロー構築
        self.tools = self._create_tools()
        self.device_ids = self.switchbot_manager.get_actual_device_ids()
        
        # ワークフローの構築
        self.workflow_builder = WorkflowBuilder(self.llm, self.tools, self.gemini_search_agent)
        self.graph = self.workflow_builder.create_agent_graph()
        self.conversation_graph = self.workflow_builder.create_conversation_graph()
        
        self._initialized = True
        
        init_total = time.time() - init_start
        logger.info(f"✅ SmartSpeakerAgent initialization completed: {init_total:.3f}s")
    
    def _create_llm(self):
        """LLMプロバイダーに応じてLLMを作成"""
        if self.llm_provider == LLMProvider.AZURE_OPENAI.value:
            return AzureChatOpenAI(
                api_version=os.getenv("AZURE_OPENAI_VERSION"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                azure_deployment=os.getenv("DEPLOYMENT_NAME"),
                temperature=0.1,
            )
        elif self.llm_provider == LLMProvider.GEMINI.value:
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-light",
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.1,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _create_tools(self):
        """ツールを作成"""
        create_tools_start = time.time()
        logger.info(f"🔧 Tool creation started")
        
        # SwitchBotツールを作成
        switchbot_tools = self.switchbot_manager.create_switchbot_tools()
        
        # Gemini検索ツールを作成
        gemini_tool_start = time.time()
        from langchain_core.tools import tool
        
        @tool
        def gemini_search(query: str) -> Dict[str, Any]:
            """Geminiの検索機能を使用してWeb検索を実行します
            
            Args:
                query: 背景と検索クエリ（質問や調べたいこと）背景を含めると検索精度も向上
            
            Returns:
                検索結果を含む辞書
            """
            try:
                result = self.gemini_search_agent.chat(query)
                
                return {
                    "response": result.response,
                    "citations": result.citations,
                    "grounding_metadata": result.grounding_metadata,
                    "success": True
                }
            except Exception as e:
                logger.error(f"Gemini検索エラー: {e}")
                return {
                    "response": f"検索中にエラーが発生しました: {str(e)}",
                    "citations": [],
                    "grounding_metadata": None,
                    "success": False
                }
        
        gemini_tool_time = time.time() - gemini_tool_start
        logger.info(f"⏱️ Gemini tool creation: {gemini_tool_time:.3f}s")
        
        # ツールを結合
        tools = switchbot_tools.copy()
        tools.append(gemini_search)
        
        # 全体の時間計測
        create_tools_total = time.time() - create_tools_start
        logger.info(f"✅ Tool creation completed: {len(tools)} tools in {create_tools_total:.3f}s")
        return tools
    
    def get_system_message(self) -> str:
        """システムメッセージを返す"""
        return """
あなたはスマートスピーカーの音声アシスタントです。
以下の機能を提供します：
1. 照明の点灯・消灯制御
2. エアコンの温度・モード・風量制御
3. 室内環境情報の確認（温度、湿度、照度）
4. Web検索による最新情報の提供

音声での対話を前提とした応答をしてください：
- 簡潔で聞き取りやすい回答
- 専門用語は避け、分かりやすい表現を使用
- 回答は2-3文程度に収める
- 数字や時間は明確に読み上げる
例：
- 「電気をつけて」→照明を点灯
- 「暑い/寒い」→適切な温度設定でエアコンを操作
- 「今の部屋の温度は？」→環境情報を取得
- 「〜について教えて」→Web検索で最新情報を提供。自身の情報にない場合は必ず検索すること

すべての応答は音声での読み上げに適した自然な日本語で行ってください。
"""
    
    async def chat_cycle(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """会話サイクルを実行（Alexaレイヤー用）- LangGraphベース"""
        # 初期状態構築
        conversation_history = {}
        if session_id not in conversation_history:
            conversation_history[session_id] = [SystemMessage(content=self.get_system_message())]
        
        messages = conversation_history[session_id].copy()
        messages.append(HumanMessage(content=user_input))
        
        initial_state: ConversationState = {
            "session_id": session_id,
            "messages": messages,
            "device_ids": self.device_ids,
            "llm_provider": self.llm_provider,
            "processing_start_time": time.time(),
            "current_task": None,
            "is_processing": False,
            "has_timeout": False,
            "partial_result": None,
            "user_input_type": "",
            "should_continue_processing": False,
            "prepared_response": "",
            "cycle_complete": False
        }
        
        try:
            # 会話サイクル専用のLangGraphを実行
            result = await self.conversation_graph.ainvoke(initial_state)
            
            return {
                "prepared_response": result.get("prepared_response", "処理が完了しました。"),
                "should_continue_processing": result.get("should_continue_processing", False)
            }
            
        except Exception as e:
            logger.error(f"Chat cycle error: {e}")
            return {
                "prepared_response": "申し訳ございません。処理中にエラーが発生しました。",
                "should_continue_processing": False
            }

    async def chat(self, user_input: str, session_id: str, conversation_history: Dict[str, List] = None) -> str:
        """ユーザー入力を処理して応答を返す"""
        start_time = time.time()
        
        try:
            if conversation_history is None:
                conversation_history = {}
            
            # セッションの会話履歴を取得または初期化
            if session_id not in conversation_history:
                conversation_history[session_id] = [SystemMessage(content=self.get_system_message())]
            
            messages = conversation_history[session_id].copy()
            messages.append(HumanMessage(content=user_input))
            
            # 初期状態を設定
            state = AgentState(
                messages=messages,
                device_ids=self.device_ids,
                llm_provider=self.llm_provider
            )
            
            # グラフを非同期実行
            graph_start = time.time()
            result = await self.graph.ainvoke(state)
            graph_time = time.time() - graph_start
            
            # 会話履歴を更新
            conversation_history[session_id] = result["messages"]
            
            # 最後のAIメッセージを取得
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                response_content = last_message.content
                
                # 実行時間のログ出力
                total_time = time.time() - start_time
                logger.info(f"⏱️ Performance Metrics - Total: {total_time:.2f}s | Graph: {graph_time:.2f}s")
                logger.info(f"SmartSpeaker-Response: {response_content}")
                return response_content
            else:
                total_time = time.time() - start_time
                logger.warning(f"⏱️ Performance Metrics - Total: {total_time:.2f}s (Failed)")
                return "申し訳ありません。応答の生成に失敗しました。"
                
        except Exception as e:
            total_time = time.time() - start_time
            error_message = f"エラーが発生しました: {str(e)}"
            logger.error(f"⏱️ Performance Metrics - Total: {total_time:.2f}s (Error)")
            logger.error(error_message)
            return "申し訳ありません。処理中にエラーが発生しました。"




# 使用例とテスト
async def main():
    """非同期メイン関数"""
    try:
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
        print(f"テスト実行エラー: {str(e)}")
        print("環境変数が設定されているか確認してください")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())