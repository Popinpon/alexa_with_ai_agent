import json
import logging
import os
import asyncio
import time
from typing import Dict, List, Any, TypedDict, Annotated
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from shared.gemini_agent import GeminiAgent

# from langchain_mcp_adapters.client import MultiServerMCPClient
from shared.switchbot import SwitchBotClient

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    device_ids: Dict[str, str]
    llm_provider: str


class SmartSpeakerAgent:
    def __init__(self, llm_provider: str = "azure_openai"):
        init_start = time.time()
        logger.info(f"🚀 SmartSpeakerAgent initialization started")
        
        self.llm_provider = llm_provider
        # SwitchBotクライアントの初期化
        self.switchbot_client = SwitchBotClient()
        # デバイス情報キャッシュ
        self._device_cache = None
        self._device_cache_timestamp = None
        self._cache_ttl = 300  # 5分間有効
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
        
        # 初期化を実行
        logger.info(f"🔄 Starting initialization during __init__")
        self.tools = self._create_tools()
        self.device_ids = self.get_actual_device_ids()
        self.graph = self._create_graph()
        self._initialized = True
        
        init_total = time.time() - init_start
        logger.info(f"✅ SmartSpeakerAgent initialization completed: {init_total:.3f}s")
        

    
    
    async def _initialize_mcp_client(self):
        """MCPクライアントを初期化（記事に従った実装）"""
        if self.mcp_client:
            return self.mcp_client
            
        try:
            mcp_start = time.time()
            logger.info(f"🔌 MCP client initialization started")
            
            # ユーザー認証情報は現在使用しない
            
            # SwitchBot MCP サーバーの設定（.mcp.jsonから取得）
            mcp_extension_key = os.getenv("MCP_EXTENSION_KEY")
            
            client = MultiServerMCPClient({
                "switchbot": {
                    "transport": "sse",
                    "url": "https://oai-alexa.azurewebsites.net/runtime/webhooks/mcp/sse",
                    "headers": {
                        "x-functions-key": mcp_extension_key
                    }
                }
            })
            
            mcp_time = time.time() - mcp_start
            logger.info(f"✅ SwitchBot MCPクライアント初期化完了: {mcp_time:.3f}s")
            return client
            
        except Exception as e:
            mcp_time = time.time() - mcp_start if 'mcp_start' in locals() else 0
            logger.error(f"❌ MCP client initialization failed after {mcp_time:.3f}s: {e}")
            return None
    
    def _is_cache_valid(self) -> bool:
        """キャッシュが有効かどうかを確認"""
        if self._device_cache is None or self._device_cache_timestamp is None:
            return False
        
        current_time = time.time()
        return (current_time - self._device_cache_timestamp) < self._cache_ttl
    
    def _get_switchbot_devices(self) -> Dict[str, Any]:
        """SwitchBotクライアントを使用してデバイス情報を取得（キャッシュ機能付き）"""
        # キャッシュが有効な場合はキャッシュから返す
        if self._is_cache_valid():
            logger.info("📋 Using cached device information")
            return self._device_cache
        
        try:
            result = self.switchbot_client.get_device_list()
            
            # レスポンスの'body'部分を取得
            if 'body' in result:
                device_data = result['body']
            else:
                device_data = result
            
            # キャッシュに保存
            if device_data:
                self._device_cache = device_data
                self._device_cache_timestamp = time.time()
                logger.info(f"💾 Device info cached")
            
            return device_data if device_data else {}
            
        except Exception as e:
            logger.error(f"SwitchBotデバイス取得エラー: {e}")
            return {}

    
    def get_actual_device_ids(self) -> Dict[str, str]:
        """実際のデバイス情報を取得（IoT操作時に使用）"""
        # SwitchBotデバイス一覧を取得（キャッシュ機能付き）
        devices_info = self._get_switchbot_devices()
        
        if not devices_info:
            logger.warning("SwitchBotデバイスが取得できませんでした。")

        device_mapping = {}
        
        # iot_agent.pyの実装に合わせてデバイスマッピング
        # Get light and aircon device IDs from infraredRemoteList
        light_device_id = next((device['deviceId'] for device in devices_info.get('infraredRemoteList', [])
                                if device['remoteType'] == 'Light'), None)
        
        aircon_device_id = next((device['deviceId'] for device in devices_info.get('infraredRemoteList', [])
                                if device['remoteType'] == 'Air Conditioner'), None)
        
        # Find Hub 2 device from deviceList
        hub2_device_id = None
        for device in devices_info.get('deviceList', []):
            if device.get('deviceType') == 'Hub 2':
                hub2_device_id = device['deviceId']
                logger.info(f"SmartSpeaker-Agent: Hub 2デバイスを検出: {device.get('deviceName', 'Unknown')} (ID: {hub2_device_id})")
                break
        
        if not hub2_device_id:
            logger.warning("警告: Hub 2デバイスが見つかりません。室内環境情報の取得ができません。")
        
        device_mapping = {
            'light_device_id': light_device_id,
            'aircon_device_id': aircon_device_id,
            'hub2_device_id': hub2_device_id
        }
        
        logger.info(f"SwitchBotデバイスマッピング: {device_mapping}")
        return device_mapping



    
    def _get_default_devices(self) -> Dict[str, str]:
        """デフォルトのデバイス設定を返す"""
        return {
            'light_device_id': "02-202403301114-45200468",
            'aircon_device_id': "02-202504191706-42866040", 
            'hub2_device_id': "C6FD9F3D1826"
        }
    
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
        """SwitchBotツールとGemini検索ツールを作成"""
        create_tools_start = time.time()
        logger.info(f"🔧 Tool creation started")
        tools = []
        
        # SwitchBotツールを作成
        switchbot_tools_start = time.time()
        
        @tool
        def get_switchbot_devices() -> Dict[str, Any]:
            """SwitchBotデバイス一覧を取得します"""
            try:
                return self.switchbot_client.get_device_list()
            except Exception as e:
                logger.error(f"SwitchBotデバイス取得エラー: {e}")
                return {"error": str(e)}
        
        @tool
        def get_device_status(device_id: str) -> Dict[str, Any]:
            """指定したSwitchBotデバイスの状態を取得します
            
            Args:
                device_id: デバイスID
            """
            try:
                return self.switchbot_client.get_device_status(device_id)
            except Exception as e:
                logger.error(f"デバイス状態取得エラー: {e}")
                return {"error": str(e)}
        
        @tool
        def control_light(device_id: str, power_state: str) -> Dict[str, Any]:
            """ライトを制御します
            
            Args:
                device_id: デバイスID
                power_state: 'on' または 'off'
            """
            try:
                return self.switchbot_client.control_light(device_id, power_state)
            except Exception as e:
                logger.error(f"ライト制御エラー: {e}")
                return {"error": str(e)}
        
        @tool
        def control_aircon(device_id: str, temperature: int, mode: int, fan_speed: int, power_state: str) -> Dict[str, Any]:
            """エアコンを制御します
            
            Args:
                device_id: デバイスID
                temperature: 温度設定（16-30）
                mode: モード（1:自動, 2:冷房, 3:除湿, 4:送風, 5:暖房）
                fan_speed: 風量（1:自動, 2:弱, 3:中, 4:強）
                power_state: 'on' または 'off'
            """
            try:
                return self.switchbot_client.control_aircon(device_id, temperature, mode, fan_speed, power_state)
            except Exception as e:
                logger.error(f"エアコン制御エラー: {e}")
                return {"error": str(e)}
        
        tools.extend([get_switchbot_devices, get_device_status, control_light, control_aircon])
        switchbot_tools_time = time.time() - switchbot_tools_start
        logger.info(f"⏱️ SwitchBot tools creation: {switchbot_tools_time:.3f}s")
        
        # Gemini検索ツール作成時間計測
        gemini_tool_start = time.time()
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
        
        tools.append(gemini_search)
        gemini_tool_time = time.time() - gemini_tool_start
        logger.info(f"⏱️ Gemini tool creation: {gemini_tool_time:.3f}s")
        
        # 全体の時間計測
        create_tools_total = time.time() - create_tools_start
        logger.info(f"✅ Tool creation completed: {len(tools)} tools in {create_tools_total:.3f}s")
        logger.info(f"📊 Breakdown - SwitchBot: {switchbot_tools_time:.3f}s | Gemini: {gemini_tool_time:.3f}s")
        return tools
    
    def _create_graph(self):
        """LangGraphのグラフを作成"""
        async def agent_node(state: AgentState):
            """エージェントノード - LLMがメッセージを処理"""
            messages = state["messages"]
            
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = await llm_with_tools.ainvoke(messages)
            
            return {"messages": [response]}
        
        async def tools_node(state: AgentState):
            """統合ツールノード - 全てのツール（MCP + Gemini）を処理"""
            messages = state["messages"]
            last_message = messages[-1]
            
            if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                return state
            
            try:
                # ツール実行時間を計測
                tool_start = time.time()
                tool_calls = last_message.tool_calls
                logger.info(f"🔧 Tool calls: {[call['name'] for call in tool_calls]}")
                
                # 統一化されたツールセット（self.tools）を使用
                tool_node = ToolNode(self.tools)
                result = await tool_node.ainvoke(state)
                
                tool_time = time.time() - tool_start
                logger.info(f"⏱️ Tool execution time: {tool_time:.2f}s")
                return result
                    
            except Exception as e:
                tool_time = time.time() - tool_start if 'tool_start' in locals() else 0
                logger.error(f"Tools node error after {tool_time:.2f}s: {e}")
                return state
        
        def should_continue(state: AgentState):
            """ツール呼び出しが必要かどうかを判定"""
            messages = state["messages"]
            last_message = messages[-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            return END
        
        # グラフを構築
        workflow = StateGraph(AgentState)
        
        # ノードを追加
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tools_node)
        
        # エッジを追加
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
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


def create_smart_speaker_agent(llm_provider: str = "azure_openai") -> SmartSpeakerAgent:
    """スマートスピーカーエージェントを作成する工場関数"""
    return SmartSpeakerAgent(llm_provider)


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
    asyncio.run(main())