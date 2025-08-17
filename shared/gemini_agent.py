"""
ChatGoogleGenerativeAIとBuilt-in Google Search toolsを使用したエージェント
"""
import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
# from google.genai import types

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()
# ログ設定
logger = logging.getLogger(__name__)



@dataclass
class SearchResult:
    """検索結果を格納するデータクラス"""
    query: str
    response: str
    citations: Optional[List[Dict[str, Any]]] = None
    grounding_metadata: Optional[Dict[str, Any]] = None

class MessageState(TypedDict):
    """GraphWorkflowで使用する状態"""
    messages: Annotated[List[Any], add_messages]

class GeminiAgent:
    """
    ChatGoogleGenerativeAIとBuilt-in Google Search toolsを使用したエージェント
    """
    
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        """
        初期化
        
        Args:
            model: 使用するGeminiモデル名
            api_key: Google AI API キー（Noneの場合は環境変数から取得）
        """
        self.model = model
        
        # APIキーの設定
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # # Google検索ツールの設定
        # self.search_tool = types.Tool(types.GoogleSearch()
        #     google_search=
        # )
        
        # # 生成設定
        # self.config = types.GenerateContentConfig(
        #     tools=[self.search_tool]
        # )
        # ChatGoogleGenerativeAIの初期化
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0.1,
            google_api_key=api_key

        )
        
        # Google Search toolを設定
        self.search_tool = GenAITool(google_search={})
        
        # システムメッセージの設定
        self.system_message = """あなたはスマートスピーカー向けの音声アシスタントです。
以下のガイドラインに従って回答してください：

3. 回答は2-3文程度に収める
5. 検索結果からの引用元を必ず組み込む。
　例：[yahoo.co.jp]なら「ヤフーによると」、titleフィールドがドメインではない場合は「Googleによると」youtubeの場合はチャンネル名や概要から発信元(テレ朝NEWSによるとなど)を推測する
6. URLなどはカタカナで表現できる場合はカタカナで表現する
7. スマートスピーカーでの読み上げに適した構成にする"""
        
        # GraphWorkflowの構築
        self.graph = self._create_graph()
        
        logger.info(f"GeminiAgent initialized with model: {self.model}")
    
    def _create_graph(self):
        """LangGraphのグラフを作成"""
        
        def search_node(state: MessageState):
            """Google Search付きのGeminiで応答を生成"""
            messages = state["messages"]
            
            # システムメッセージを追加
            full_messages = [SystemMessage(content=self.system_message)] + messages
            
            # Google Search toolを使用してクエリを実行
            response = self.llm.invoke(
                full_messages,
                tools=[self.search_tool]
            )
            
            
            return {"messages": [response]}
        
        # グラフを構築
        workflow = StateGraph(MessageState)
        workflow.add_node("search", search_node)
        workflow.add_edge(START, "search")
        workflow.add_edge("search", END)
        
        return workflow.compile()
    
    def chat(self, query: str) -> SearchResult:
        """
        チャット実行
        
        Args:
            query: ユーザーの質問
            
        Returns:
            SearchResult: 回答結果オブジェクト
        """
        try:
            logger.info(f"Executing Gemini search: {query}")
            
            # GraphWorkflowでクエリを実行
            result = self.graph.invoke({
                "messages": [HumanMessage(content=query)]

            })
            
            # 最後のメッセージから回答を取得
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                response_text = last_message.content
            else:
                response_text = "申し訳ありません。応答の生成に失敗しました。"
            
            search_result = SearchResult(
                query=query,
                response=response_text,
                citations=None,
                grounding_metadata=None
            )
            
            logger.info("Gemini search completed successfully")
            return search_result
            
        except Exception as e:
            logger.error(f"Error during Gemini search: {str(e)}")
            return SearchResult(
                query=query,
                response=f"検索エラーが発生しました: {str(e)}",
                citations=None,
                grounding_metadata=None
            )
    
    def clear_conversation(self):
        """会話履歴をクリア（この実装では状態を持たないため、何もしない）"""
        pass
    
    def get_conversation_length(self) -> int:
        """会話履歴の長さを取得（この実装では常に0）"""
        return 0
    
    def get_model_info(self) -> Dict[str, str]:
        """使用中のモデル情報を取得"""
        return {
            "model": self.model,
            "type": "gemini_agent",
            "search_tool": "google_search_builtin"
        }

# 下位互換性のためのエイリアス
LangChainGeminiAgent = GeminiAgent

# 使用例とテスト
if __name__ == "__main__":
    try:
        agent = GeminiAgent()

        # テストクエリ
        queries = [
            "今日のニュースは？",
        ]
        # response=agent.llm.invoke(queries[0],tools=[GenAITool(google_search={})])
        # print(response)
        for query in queries:
            print(f"\n=== 質問: {query} ===")
            result = agent.chat(query)
            print(f"回答: {result.response}")
            print(f"モデル情報: {agent.get_model_info()}")
            
    except Exception as e:
        print(f"テスト実行エラー: {str(e)}")
        print("環境変数GEMINI_API_KEYが設定されているか確認してください")