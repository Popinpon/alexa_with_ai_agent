"""
会話サイクル管理クラス
"""
import time
import logging
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from shared.types import ConversationState, ContinuationState

logger = logging.getLogger(__name__)

# グローバル状態管理
continuation_states: Dict[str, ContinuationState] = {}
CONTINUATION_TTL = 600  # 10分


class ConversationManager:
    """会話の継続処理とタイムアウト管理を行うクラス"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def analyze_continuation_intent(self, user_input: str) -> str:
        """LLMを使って継続意図を分析"""
        system_prompt = """
あなたは対話システムの継続判定を行います。
ユーザーの発言を分析して、以下のいずれかを判定してください：

1. "continuation_yes" - 前の処理の続きを聞きたい
2. "continuation_no" - 別の新しい質問をしたい

継続要求の例: はい、続き、お願いします、続けて
新規質問の例: いいえ、違う話、今日の天気は？、エアコンつけて

回答は "continuation_yes" または "continuation_no" のみで答えてください。
"""
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"ユーザー発言: 「{user_input}」")
            ])
            
            result = response.content.strip().lower()
            if "continuation_yes" in result:
                return "continuation_yes"
            else:
                return "continuation_no"
                
        except Exception as e:
            logger.error(f"継続判定エラー: {e}")
            return "continuation_no"  # エラー時は新規質問として処理
    
    def has_pending_session(self, session_id: str) -> bool:
        """セッションに未完了タスクがあるかチェック"""
        return session_id in continuation_states
    
    def get_continuation_state(self, session_id: str) -> Optional[ContinuationState]:
        """継続状態を取得"""
        return continuation_states.get(session_id)
    
    def save_continuation_state(self, state: ConversationState):
        """継続処理が必要な場合は状態を保存"""
        session_id = state["session_id"]
        
        continuation_state: ContinuationState = {
            "session_id": session_id,
            "original_query": state["messages"][-1].content if state["messages"] else "",
            "current_task": state.get("current_task", "処理中"),
            "partial_result": state.get("partial_result", ""),
            "saved_at": time.time(),
            "conversation_state": state
        }
        
        continuation_states[session_id] = continuation_state
        logger.info(f"継続状態を保存: {session_id}")
    
    def cleanup_expired_states(self):
        """期限切れの継続状態をクリーンアップ"""
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, state in continuation_states.items()
            if (current_time - state["saved_at"]) > CONTINUATION_TTL
        ]
        
        for session_id in expired_sessions:
            del continuation_states[session_id]
            logger.info(f"期限切れ継続状態を削除: {session_id}")
    
    def prepare_timeout_response(self, state: ConversationState) -> Dict[str, Any]:
        """タイムアウト時の部分応答を準備"""
        task = state.get("current_task", "処理中")
        response = f"{task}です。まだ少し時間がかかりそうです。続きを聞きますか？"
        return {
            "prepared_response": response,
            "should_continue_processing": True,
            "cycle_complete": True  # ユーザー応答待ち
        }
    
    def prepare_completion_response(self, state: ConversationState) -> Dict[str, Any]:
        """完了時の最終応答を準備"""
        final_response = state.get("partial_result", "処理が完了しました。")
        return {
            "prepared_response": final_response,
            "should_continue_processing": False,
            "cycle_complete": True
        }