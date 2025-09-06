"""
Smart Speaker Agent の型定義とEnum
"""
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from enum import Enum
from langgraph.graph.message import add_messages


class LLMProvider(Enum):
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    device_ids: Dict[str, str]
    llm_provider: str


class ConversationState(TypedDict):
    # 基本情報
    session_id: str
    messages: Annotated[List[Any], add_messages]
    device_ids: Dict[str, str]
    llm_provider: str
    
    # 処理状態
    processing_start_time: Optional[float]
    current_task: Optional[str]
    is_processing: bool
    
    # タイムアウト管理
    has_timeout: bool
    partial_result: Optional[str]
    
    # サイクル制御
    user_input_type: str  # "new_question" | "continuation_yes" | "continuation_no"
    should_continue_processing: bool
    
    # 応答準備
    prepared_response: str
    cycle_complete: bool


class ContinuationState(TypedDict):
    """継続処理用の軽量な状態"""
    session_id: str
    original_query: str
    current_task: str
    partial_result: str
    saved_at: float
    conversation_state: ConversationState