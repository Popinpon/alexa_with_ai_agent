import json
import logging
import os
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .switchbot import control_light, control_aircon, get_device_list, get_device_status


class LLMProvider(Enum):
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    device_ids: Dict[str, str]
    llm_provider: str


class IoTAgent:
    def __init__(self, token: str, secret: str, llm_provider: str = "azure_openai"):
        self.token = token
        self.secret = secret
        self.llm_provider = llm_provider
        self.device_ids = self._initialize_devices()
        self.llm = self._create_llm()
        self.tools = self._create_tools()
        self.graph = self._create_graph()
    
    def _initialize_devices(self) -> Dict[str, str]:
        """デバイス情報を初期化する"""
        try:
            device_list = get_device_list(self.token, self.secret)
            
            light_device_id = next((device['deviceId'] for device in device_list['body']['infraredRemoteList'] 
                                   if device['remoteType'] == 'Light'), None)
            
            aircon_device_id = next((device['deviceId'] for device in device_list['body']['infraredRemoteList'] 
                                    if device['remoteType'] == 'Air Conditioner'), None)
            
            hub2_device_id = None
            for device in device_list['body'].get('deviceList', []):
                if device.get('deviceType') == 'Hub 2':
                    hub2_device_id = device['deviceId']
                    logging.info(f"IoT-Agent: Hub 2デバイスを検出: {device['deviceName']} (ID: {hub2_device_id})")
                    break
            
            if not hub2_device_id:
                logging.warning("警告: Hub 2デバイスが見つかりません。室内環境情報の取得ができません。")
            
            return {
                'light_device_id': light_device_id,
                'aircon_device_id': aircon_device_id,
                'hub2_device_id': hub2_device_id
            }
        except Exception as e:
            logging.error(f"デバイス初期化エラー: {str(e)}")
            return {
                'light_device_id': None,
                'aircon_device_id': None,
                'hub2_device_id': None
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
                model="gemini-1.5-flash",
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.1,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _create_tools(self):
        """ツールを作成"""
        @tool
        def light_control(power_state: str) -> Dict[str, Any]:
            """電気を点けたり消したりします
            
            Args:
                power_state: 電気のオン/オフ状態。onかoffのどちらかを指定してください。
            """
            device_id = self.device_ids.get('light_device_id')
            if not device_id:
                return {"error": "ライトデバイスが見つかりません"}
            
            result = control_light(device_id, self.token, self.secret, power_state)
            logging.info(f"IoT-Agent: ライト{power_state}制御実行")
            return result
        
        @tool
        def aircon_control(power_state: str, temp: int = 24, mode: int = 2, fan_speed: int = 3) -> Dict[str, Any]:
            """エアコンを操作します
            
            Args:
                power_state: エアコンのオン/オフ状態。onかoffのどちらかを指定してください。
                temp: 設定温度（摂氏、16〜30度）
                mode: 運転モード: 0/1=自動, 2=冷房, 3=除湿, 4=送風, 5=暖房
                fan_speed: 風量: 1=自動, 2=弱, 3=中, 4=強
            """
            device_id = self.device_ids.get('aircon_device_id')
            if not device_id:
                return {"error": "エアコンデバイスが見つかりません"}
            
            result = control_aircon(device_id, self.token, self.secret, temp, mode, fan_speed, power_state)
            logging.info(f"IoT-Agent: エアコン{power_state}制御実行: 温度={temp}度, モード={mode}, 風量={fan_speed}")
            return result
        
        @tool
        def get_room_environment() -> Dict[str, Any]:
            """室内の環境情報（温度、湿度、照度）を取得します"""
            device_id = self.device_ids.get('hub2_device_id')
            if not device_id:
                return {
                    "error": "Hub 2デバイスが見つかりません。室内環境情報を取得できません。"
                }
            
            hub_status = get_device_status(device_id, self.token, self.secret)
            
            if hub_status['statusCode'] != 100:
                return {
                    "error": f"環境情報の取得に失敗しました: {hub_status.get('message', '不明なエラー')}"
                }
            
            status_data = hub_status['body']
            environment_info = {
                "temperature": status_data.get('temperature', None),
                "humidity": status_data.get('humidity', None),
                "lightLevel": status_data.get('lightLevel', None),
            }
            
            temp = environment_info.get("temperature")
            humidity = environment_info.get("humidity")
            light = environment_info.get("lightLevel")
            logging.info(f"IoT-Agent: 室内環境情報取得: 温度={temp}°C, 湿度={humidity}%, 照度={light}/20")
            
            return environment_info
        
        return [light_control, aircon_control, get_room_environment]
    
    def _create_graph(self):
        """LangGraphのグラフを作成"""
        def agent_node(state: AgentState):
            """エージェントノード - LLMがメッセージを処理"""
            messages = state["messages"]
            
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = llm_with_tools.invoke(messages)
            
            return {"messages": [response]}
        
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
        workflow.add_node("tools", ToolNode(self.tools))
        
        # エッジを追加
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def get_system_message(self) -> str:
        """システムメッセージを返す"""
        return """
あなたはIoT家電を操作できるAIアシスタントです。
以下の機能を提供します：
1. 電気のオン/オフを制御
2. エアコンの温度・モード・風量を制御
3. 室内の環境情報（温度、湿度、照度）を取得

ユーザーの要望を理解して、適切なIoTデバイスを操作してください。
ユーザーが暑い/寒いと言った場合は、適切な温度設定でエアコンを操作してください。
「電気をつけて」と言われたら、light_control関数で電気をオンにします。
「今の室温は？」や「部屋の温度を教えて」などと言われたら、get_room_environment関数で環境情報を取得します。
会話は日本語で行い、音声アシスタント用に短く、読み上げやすい文章で応答してください。
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
                logging.info(f"AI-Response: {response_content}")
                return response_content
            else:
                return "申し訳ありません。応答の生成に失敗しました。"
                
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            logging.error(error_message)
            return "申し訳ありません。処理中にエラーが発生しました。"


def create_iot_agent(token: str, secret: str, llm_provider: str = "azure_openai") -> IoTAgent:
    """IoTエージェントを作成する工場関数"""
    return IoTAgent(token, secret, llm_provider)