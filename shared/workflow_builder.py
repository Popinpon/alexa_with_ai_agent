"""
LangGraph ワークフロー構築クラス
"""
import time
import asyncio
import logging
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from shared.types import AgentState, ConversationState
from shared.conversation_manager import ConversationManager
from shared.gemini_agent import GeminiAgent

logger = logging.getLogger(__name__)


class WorkflowBuilder:
    """LangGraph ワークフローの構築を行うクラス"""
    
    def __init__(self, llm, tools: List, gemini_agent: GeminiAgent):
        self.llm = llm
        self.tools = tools
        self.gemini_agent = gemini_agent
        self.conversation_manager = ConversationManager(llm)
    
    def create_gemini_tool(self):
        """Gemini検索ツールを作成"""
        @tool
        def gemini_search(query: str) -> Dict[str, Any]:
            """Geminiの検索機能を使用してWeb検索を実行します
            
            Args:
                query: 背景と検索クエリ（質問や調べたいこと）背景を含めると検索精度も向上
            
            Returns:
                検索結果を含む辞書
            """
            try:
                result = self.gemini_agent.chat(query)
                
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
        
        return gemini_search
    
    def create_agent_graph(self):
        """通常のchat用のグラフを作成"""
        
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
        
        # AgentState用のグラフ（通常のchat）
        agent_workflow = StateGraph(AgentState)
        agent_workflow.add_node("agent", agent_node)
        agent_workflow.add_node("tools", tools_node)
        agent_workflow.add_edge(START, "agent")
        agent_workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        agent_workflow.add_edge("tools", "agent")
        
        return agent_workflow.compile()
    
    def create_conversation_graph(self):
        """会話サイクル用のグラフを作成"""
        
        async def input_analysis_node(state: ConversationState):
            """入力分析ノード"""
            last_message = state["messages"][-1].content
            session_id = state["session_id"]
            
            # セッションに未完了タスクがあるかチェック
            has_pending = self.conversation_manager.has_pending_session(session_id)
            
            if not has_pending:
                input_type = "new_question"
                processing_start = time.time()
            else:
                # LLMで継続意図を判定
                input_type = await self.conversation_manager.analyze_continuation_intent(last_message)
                processing_start = state.get("processing_start_time", time.time())
            
            return {
                "user_input_type": input_type,
                "processing_start_time": processing_start,
                "is_processing": True,
                "has_timeout": False
            }
        
        async def agent_processing_node(state: ConversationState):
            """エージェント処理ノード（タイムアウト監視付き）"""
            input_type = state["user_input_type"]
            
            if input_type == "continuation_yes":
                return await self._resume_processing(state)
            else:
                return await self._process_with_timeout_monitoring(state)
        
        async def response_preparation_node(state: ConversationState):
            """応答準備ノード"""
            if state["has_timeout"]:
                return self.conversation_manager.prepare_timeout_response(state)
            else:
                return self.conversation_manager.prepare_completion_response(state)
        
        async def state_management_node(state: ConversationState):
            """状態管理ノード"""
            if state["should_continue_processing"]:
                self.conversation_manager.save_continuation_state(state)
            
            return {"cycle_complete": True}
        
        def should_continue_conversation(state: ConversationState):
            """会話サイクルを続けるかどうかを判定"""
            if state.get("cycle_complete", False):
                return END
            return "response_preparation"
        
        # ConversationState用のグラフ（会話サイクル）
        conversation_workflow = StateGraph(ConversationState)
        conversation_workflow.add_node("input_analysis", input_analysis_node)
        conversation_workflow.add_node("agent_processing", agent_processing_node)
        conversation_workflow.add_node("response_preparation", response_preparation_node)
        conversation_workflow.add_node("state_management", state_management_node)
        conversation_workflow.add_edge(START, "input_analysis")
        conversation_workflow.add_edge("input_analysis", "agent_processing")
        conversation_workflow.add_edge("agent_processing", "response_preparation")
        conversation_workflow.add_edge("response_preparation", "state_management")
        conversation_workflow.add_conditional_edges("state_management", should_continue_conversation, ["response_preparation", END])
        
        return conversation_workflow.compile()
    
    async def _resume_processing(self, state: ConversationState):
        """継続処理を実行"""
        session_id = state["session_id"]
        
        saved_state = self.conversation_manager.get_continuation_state(session_id)
        if saved_state:
            # 保存された結果を返す（実際の実装では処理を再開）
            return {
                "current_task": saved_state["current_task"],
                "partial_result": saved_state["partial_result"],
                "is_processing": False,
                "has_timeout": False
            }
        else:
            return {
                "current_task": "継続処理",
                "partial_result": "継続する処理が見つかりませんでした",
                "is_processing": False,
                "has_timeout": False
            }
    
    async def _process_with_timeout_monitoring(self, state: ConversationState):
        """タイムアウト監視付き処理"""
        TIMEOUT_SECONDS = 7.0  # 安全マージン
        processing_start = state["processing_start_time"]
        
        # AgentStateを構築
        agent_state = AgentState(
            messages=state["messages"],
            device_ids=state["device_ids"],
            llm_provider=state["llm_provider"]
        )
        
        try:
            # エージェントグラフを作成（必要に応じて）
            agent_graph = self.create_agent_graph()
            
            # タイムアウト付きでグラフ実行
            result = await asyncio.wait_for(
                agent_graph.ainvoke(agent_state),
                timeout=TIMEOUT_SECONDS
            )
            
            # 成功時の処理
            final_response = self._extract_agent_response(result)
            return {
                "current_task": "処理完了",
                "partial_result": final_response,
                "is_processing": False,
                "has_timeout": False
            }
            
        except asyncio.TimeoutError:
            # タイムアウト時の処理
            elapsed = time.time() - processing_start
            return {
                "current_task": "情報を取得中",
                "partial_result": f"処理中です（{elapsed:.1f}秒経過）",
                "is_processing": True,
                "has_timeout": True
            }
    
    def _extract_agent_response(self, result):
        """エージェント実行結果から応答を抽出"""
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                return last_message.content
        return "処理が完了しました。"