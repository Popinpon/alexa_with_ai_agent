import json
import logging
import os
import asyncio
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from .gemini_search_agent import GeminiSearchAgent

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("langchain-mcp not available. Falling back to direct switchbot integration.")


class LLMProvider(Enum):
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    device_ids: Dict[str, str]
    llm_provider: str


class SmartSpeakerAgent:
    def __init__(self, llm_provider: str = "azure_openai"):
        self.llm_provider = llm_provider
        self.mcp_client = None
        self.device_ids = self._initialize_devices()
        self.llm = self._create_llm()
        self.gemini_search_agent = GeminiSearchAgent()
        self.tools = asyncio.run(self._create_tools())
        self.graph = self._create_graph()
    
    async def _initialize_mcp_client(self):
        """MCPクライアントを初期化（記事に従った実装）"""
        if not MCP_AVAILABLE:
            return None
            
        try:
            # ユーザー認証情報は現在使用しない
            
            # SwitchBot MCP サーバーの設定（.mcp.jsonから取得）
            import os
            mcp_extension_key = os.getenv("MCP_EXTENSION_KEY")
            
            client = MultiServerMCPClient({
                "switchbot": {
                    "type": "sse",
                    "url": "https://oai-alexa.azurewebsites.net/runtime/webhooks/mcp/sse",
                    "headers": {
                        "x-functions-key": mcp_extension_key
                    }
                }
            })
            
            logging.info("SwitchBot MCPクライアントを初期化しました")
            return client
            
        except Exception as e:
            logging.error(f"MCP client initialization failed: {e}")
            return None
    
    def _initialize_devices(self) -> Dict[str, str]:
        """SwitchBot MCPツールからデバイス情報を動的に取得"""
        try:
            # SwitchBotデバイス一覧を取得
            devices_info = self._get_switchbot_devices()
            
            if not devices_info:
                logging.warning("SwitchBotデバイスが取得できませんでした。デフォルト設定を使用します。")
                return self._get_default_devices()
            
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
                    logging.info(f"SmartSpeaker-Agent: Hub 2デバイスを検出: {device.get('deviceName', 'Unknown')} (ID: {hub2_device_id})")
                    break
            
            if not hub2_device_id:
                logging.warning("警告: Hub 2デバイスが見つかりません。室内環境情報の取得ができません。")
            
            device_mapping = {
                'light_device_id': light_device_id,
                'aircon_device_id': aircon_device_id,
                'hub2_device_id': hub2_device_id
            }
            
            logging.info(f"SwitchBotデバイスマッピング: {device_mapping}")
            return device_mapping if device_mapping else self._get_default_devices()
            
        except Exception as e:
            logging.error(f"デバイス情報の取得に失敗: {e}")
            return self._get_default_devices()

    
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
    
    async def _create_tools(self):
        """MCPクライアントからSwitchBotツールとGemini検索ツールを取得"""
        tools = []
        
        # MCPクライアントの初期化
        self.mcp_client = await self._initialize_mcp_client()
        
        if self.mcp_client and MCP_AVAILABLE:
            try:
                # MCPクライアントからツールを取得
                mcp_tools = await self.mcp_client.get_tools()
                logging.info(f"SwitchBot MCPツールを取得: {len(mcp_tools)}個")
                tools.extend(mcp_tools)
            except Exception as e:
                logging.error(f"MCPツールの取得に失敗: {e}")
        else:
            logging.warning("MCPツールが利用できません")
        
        # Gemini検索ツールを追加
        @tool
        def gemini_search(query: str) -> Dict[str, Any]:
            """Geminiの検索機能を使用してWeb検索を実行します
            
            Args:
                query: 検索クエリ（質問や調べたいこと）
            
            Returns:
                検索結果を含む辞書
            """
            try:
                result = self.gemini_search_agent.chat(query, add_citations=True)
                
                return {
                    "response": result.response,
                    "citations": result.citations,
                    "grounding_metadata": result.grounding_metadata,
                    "success": True
                }
            except Exception as e:
                logging.error(f"Gemini検索エラー: {e}")
                return {
                    "response": f"検索中にエラーが発生しました: {str(e)}",
                    "citations": [],
                    "grounding_metadata": None,
                    "success": False
                }
        
        tools.append(gemini_search)
        logging.info(f"全ツール数: {len(tools)}個（SwitchBot + Gemini検索）")
        return tools
    
    def _create_graph(self):
        """LangGraphのグラフを作成（MCPツールのみ使用）"""
        def agent_node(state: AgentState):
            """エージェントノード - LLMがメッセージを処理"""
            messages = state["messages"]
            
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = llm_with_tools.invoke(messages)
            
            return {"messages": [response]}
        
        async def mcp_tools_node(state: AgentState):
            """MCP ツールノード - 記事に従った実装"""
            messages = state["messages"]
            last_message = messages[-1]
            
            if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                return state
            
            try:
                if self.mcp_client and MCP_AVAILABLE:
                    # MCPツールを使用してツール呼び出しを実行
                    mcp_tools = await self.mcp_client.get_tools()
                    tool_node = ToolNode(mcp_tools)
                    result = tool_node.invoke(state)
                    return result
                else:
                    logging.error("MCPツールが利用できません")
                    return state
                    
            except Exception as e:
                logging.error(f"MCP tools node error: {e}")
                return state
        
        def should_continue(state: AgentState):
            """ツール呼び出しが必要かどうかを判定"""
            messages = state["messages"]
            last_message = messages[-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "mcp_tools"
            return END
        
        # グラフを構築
        workflow = StateGraph(AgentState)
        
        # ノードを追加
        workflow.add_node("agent", agent_node)
        workflow.add_node("mcp_tools", mcp_tools_node)
        
        # エッジを追加
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, ["mcp_tools", END])
        workflow.add_edge("mcp_tools", "agent")
        
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
- 「電気をつけて」→照明を点灯
- 「暑い/寒い」→適切な温度設定でエアコンを操作
- 「今の部屋の温度は？」→環境情報を取得
- 「〜について教えて」→Web検索で最新情報を提供

すべての応答は音声での読み上げに適した自然な日本語で行ってください。
"""
    
    def chat(self, user_input: str, session_id: str, conversation_history: Dict[str, List] = None) -> str:
        """ユーザー入力を処理して応答を返す"""
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
        
        try:
            # グラフを実行
            result = self.graph.invoke(state)
            
            # 会話履歴を更新
            conversation_history[session_id] = result["messages"]
            
            # 最後のAIメッセージを取得
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                response_content = last_message.content
                logging.info(f"SmartSpeaker-Response: {response_content}")
                return response_content
            else:
                return "申し訳ありません。応答の生成に失敗しました。"
                
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            logging.error(error_message)
            return "申し訳ありません。処理中にエラーが発生しました。"


def create_smart_speaker_agent(llm_provider: str = "azure_openai") -> SmartSpeakerAgent:
    """スマートスピーカーエージェントを作成する工場関数"""
    return SmartSpeakerAgent(llm_provider)