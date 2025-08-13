"""
GeminiのWeb検索（Grounding with Google Search）を利用したエージェントクラス
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

# ログ設定
logger = logging.getLogger(__name__)
import pickle
import logging
logging.basicConfig(level=logging.INFO)

# google-genaiパッケージが利用可能かチェック
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-genai package not found. Please install: pip install google-genai")


@dataclass
class SearchResult:
    """検索結果を格納するデータクラス"""
    query: str
    response: str
    citations: list
    grounding_metadata: Optional[Dict[str, Any]] = None


class GeminiSearchAgent:
    @staticmethod
    def investigate_text_alignment(content_text, segments, api_indices=None):
        """
        segment.textとcontent.textの一致箇所・前後の文字・改行や空白の有無を調査
        APIのstart_index/end_index（バイト単位）で正しく切り出せるかも検証
        api_indices: [(start_index, end_index), ...]  # バイト単位
        """
        def byte_index_to_str_index(text, byte_index):
            encoded = text.encode('utf-8')
            if byte_index >= len(encoded):
                return len(text)
            sub = encoded[:byte_index]
            return len(sub.decode('utf-8', errors='ignore'))

        for idx, seg in enumerate(segments):

            # API index検証
            if api_indices and idx < len(api_indices):
                api_start, api_end = api_indices[idx]
                str_start = byte_index_to_str_index(content_text, api_start)
                str_end = byte_index_to_str_index(content_text, api_end)
                api_slice = content_text[str_start:str_end]
                print(f"API index: start={api_start} end={api_end} (str_start={str_start} str_end={str_end})")
                print(f"APIで切り出した: '{api_slice}'")
                print(f"API index一致: {api_slice == seg}")
            print("-"*40)
    """
    Gemini API のGoogle Search Grounding機能を利用した検索エージェント
    """
    @staticmethod
    def save_response(response, filename):
        """Gemini APIのresponseオブジェクトをpickleで保存"""
        with open(filename, "wb") as f:
            pickle.dump(response, f)

    @staticmethod
    def load_response(filename):
        """pickleで保存したresponseオブジェクトをロード"""
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    def __init__(self, model: str = "gemini-2.5-flash-light", api_key: Optional[str] = None):
        """
        GeminiSearchAgentの初期化
        
        Args:
            model: 使用するGeminiモデル名
            api_key: Gemini API キー（Noneの場合は環境変数から取得）
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai package is required. Install with: pip install google-genai")
        
        self.model = model
        
        # APIキーの設定
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GEMINI_API_KEY")
            
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter is required")
        
        # Geminiクライアントの初期化
        self.client = genai.Client(api_key=self.api_key)
        
        # Google検索ツールの設定
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        # 生成設定
        self.config = types.GenerateContentConfig(
            tools=[self.grounding_tool]
        )
        
        # 会話履歴の初期化（スマートスピーカー向けシステムプロンプトを含む）
        system_prompt = """あなたはスマートスピーカー向けの音声アシスタントです。
以下のガイドラインに従って回答してください：

1. 簡潔で聞き取りやすい回答を心がける
2. 専門用語は避け、一般的な表現を使用する
3. 回答は2-3文程度に収める
4. 数字や時間は明確に伝える
5. 引用情報は自然な日本語で組み込む
6. urlなどはカタカナで表現できる場合はカタカナで表現する
7. スマートスピーカーでの読み上げに適した構成にする"""

        self.conversation_history = [
            types.Content(
                role="user",
                parts=[types.Part(text=system_prompt)]
            ),
            types.Content(
                role="model", 
                parts=[types.Part(text="承知しました。スマートスピーカー向けに簡潔で聞き取りやすい回答を提供します。")]
            )
        ]
        
        logger.info(f"GeminiSearchAgent initialized with model: {self.model}")
    
    def chat(self, query: str, add_citations: bool = False, save_raw_response: bool = False, raw_response_path: str = "test_gemini_response.pkl") -> SearchResult:
        """
        会話履歴を保持しながらGemini APIのGoogle Search Groundingを使用してチャット
        
        Args:
            query: ユーザーの質問
            add_citations: レスポンステキストに引用を追加するか
            save_raw_response: TrueならGemini APIの生responseをpickle保存
            raw_response_path: 保存ファイルパス
            
        Returns:
            SearchResult: 回答結果オブジェクト
        """
        try:
            logger.info(f"Executing grounded search: {query}")
            
            # 会話履歴に新しいユーザーメッセージを追加
            self.conversation_history.append(types.Content(
                role="user",
                parts=[types.Part(text=query)]
            ))
            
            # Gemini APIでGoogle Search Groundingを使用（会話履歴を含む）
            response = self.client.models.generate_content(
                model=self.model,
                contents=self.conversation_history,
                config=self.config
            )
            # 生responseを保存
            if save_raw_response:
                self.save_response(response, raw_response_path)
            
            # 基本的なレスポンステキスト
            response_text = response.text
            citations = []
            grounding_metadata = None
            
            # グラウンディングメタデータが存在する場合
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    print(candidate.grounding_metadata)
                    grounding_metadata = self._extract_grounding_metadata(candidate.grounding_metadata)
                    print(grounding_metadata)
                    
                    if add_citations:
                        response_text, citations = self._add_citations_to_text(
                            response_text, candidate.grounding_metadata
                        )
            
            # アシスタントの回答を会話履歴に追加
            self.conversation_history.append(types.Content(
                role="model",
                parts=[types.Part(text=response_text)]
            ))
            
            result = SearchResult(
                query=query,
                response=response_text,
                citations=citations,
                grounding_metadata=grounding_metadata
            )
            
            logger.info("Search completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error during grounded search: {str(e)}")
            return SearchResult(
                query=query,
                response=f"検索エラーが発生しました: {str(e)}",
                citations=[],
                grounding_metadata=None
            )
    
    def _extract_grounding_metadata(self, metadata) -> Dict[str, Any]:
        """グラウンディングメタデータを辞書形式で抽出"""
        try:
            result = {
                "web_search_queries": [],  # Gemini APIドキュメント通り
                "web_results": []
            }
            
            # Gemini APIリファレンス通り、web_search_queriesを取得
            # web_search_queriesを取得
            if hasattr(metadata, 'web_search_queries'):
                result["web_search_queries"] = metadata.web_search_queries
            
            # groundingChunksからweb結果を取得
            if hasattr(metadata, 'grounding_chunks') or hasattr(metadata, 'groundingChunks'):
                chunks = getattr(metadata, 'grounding_chunks', None) or getattr(metadata, 'groundingChunks', [])
                for chunk in chunks:
                    if hasattr(chunk, 'web') and chunk.web:
                        result["web_results"].append({
                            "title": getattr(chunk.web, 'title', ''),
                            "uri": getattr(chunk.web, 'uri', ''),
                            "summary": getattr(chunk.web, 'summary', '')
                        })
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to extract grounding metadata: {str(e)}")
            return {}
    
    def _add_citations_to_text(self, text: str, metadata) -> tuple[str, list]:
        """グラウンディングメタデータを基にテキストに簡素な引用を追加"""
        try:
            citations = []
            if not hasattr(metadata, 'grounding_supports') or not hasattr(metadata, 'grounding_chunks'):
                return text, citations

            supports = metadata.grounding_supports
            chunks = metadata.grounding_chunks
            
            # バイトインデックスを文字インデックスに変換する関数
            def byte_index_to_str_index(text, byte_index):
                encoded = text.encode('utf-8')
                if byte_index >= len(encoded):
                    return len(text)
                sub = encoded[:byte_index]
                return len(sub.decode('utf-8', errors='ignore'))
            
            # 引用情報を収集（titleのみの簡素な形式）
            citation_titles = []
            for chunk in chunks:
                if hasattr(chunk, 'web') and chunk.web:
                    title = getattr(chunk.web, 'title', '')
                    if title and title not in citation_titles:
                        citation_titles.append(title)
                        citations.append({
                            'index': len(citation_titles),
                            'title': title
                        })
            
            # テキストに引用を挿入（文の始まりに挿入）
            modified_text = text
            insertions = []
            
            for support in supports:
                if hasattr(support, 'segment') and support.segment:
                    segment = support.segment
                    start_byte = getattr(segment, 'start_index', 0)
                    end_byte = getattr(segment, 'end_index', 0)
                    
                    # バイトインデックスを文字インデックスに変換
                    start_idx = byte_index_to_str_index(text, start_byte)
                    
                    # 引用元のタイトルを取得（複数の場合は組み合わせる）
                    chunk_indices = getattr(support, 'grounding_chunk_indices', [])
                    if chunk_indices:
                        titles = []
                        for chunk_idx in chunk_indices:
                            if chunk_idx < len(chunks):
                                chunk = chunks[chunk_idx]
                                if hasattr(chunk, 'web') and chunk.web:
                                    title = getattr(chunk.web, 'title', '')
                                    if title and title not in titles:
                                        titles.append(title)
                        
                        if titles:
                            def is_domain_title(title):
                                # ドメイン名らしいタイトルかチェック（.comや.jpを含む、短い）
                                return '.' in title and len(title.split()) <= 2
                            
                            def format_title(title):
                                # ドメインでない場合はGoogleの情報として表示
                                if is_domain_title(title):
                                    return f"{title}によると、"
                                else:
                                    return "Googleの情報によると、"
                            
                            if len(titles) == 1:
                                citation_text = format_title(titles[0])
                            else:
                                # 複数の場合、ドメインが含まれているかチェック
                                has_domain = any(is_domain_title(t) for t in titles)
                                if has_domain:
                                    domain_title = next(t for t in titles if is_domain_title(t))
                                    citation_text = f"{domain_title}など複数の情報源によると、"
                                else:
                                    citation_text = "Googleなど複数の情報源によると、"
                            insertions.append((start_idx, citation_text))
            
            # 後ろから挿入して位置がずれないようにする
            for pos, citation_text in sorted(insertions, reverse=True):
                modified_text = modified_text[:pos] + citation_text + modified_text[pos:]

            return modified_text, citations
        except Exception as e:
            logger.warning(f"Failed to add citations: {str(e)}")
            return text, []
    
    def clear_conversation(self):
        """会話履歴をクリア（システムプロンプトは保持）"""
        system_prompt = """あなたはスマートスピーカー向けの音声アシスタントです。
以下のガイドラインに従って回答してください：

1. 簡潔で聞き取りやすい回答を心がける
2. 専門用語は避け、一般的な表現を使用する
3. 回答は2-3文程度に収める
4. 数字や時間は明確に伝える
5. 引用情報は自然な日本語で組み込む
6. スマートスピーカーでの読み上げに適した構成にする"""

        self.conversation_history = [
            types.Content(
                role="user",
                parts=[types.Part(text=system_prompt)]
            ),
            types.Content(
                role="model", 
                parts=[types.Part(text="承知しました。スマートスピーカー向けに簡潔で聞き取りやすい回答を提供します。")]
            )
        ]
        logger.info("Conversation history cleared (system prompt retained)")
    
    def get_conversation_length(self) -> int:
        """会話履歴の長さを取得"""
        return len(self.conversation_history)
    
    def get_model_info(self) -> Dict[str, str]:
        """使用中のモデル情報を取得"""
        return {
            "model": self.model,
            "api_available": str(GEMINI_AVAILABLE),
            "conversation_length": str(self.get_conversation_length())
        }


# 使用例（テスト用）
if __name__ == "__main__":
    try:
        agent = GeminiSearchAgent()
        TEST_MODE = os.getenv("GEMINI_TEST_MODE", "0") == "0"
        response_file = "test_gemini_response.pkl"
        print(f"TEST_MODE: {TEST_MODE}")
        
        # 会話形式のテスト
        queries = [
            "大阪で時間ごとの降水確率を教えて",

        ]
        
        for i, query in enumerate(queries):
            print(f"\n=== 質問 {i+1} ===")
            print(f"ユーザー: {query}")
            
            if TEST_MODE and i == 0 and os.path.exists(response_file):
                # 最初の質問のみ保存済みresponseをロード
                response = agent.load_response(response_file)
                print("gemini response",response)
                print("=====")
                candidate = response.candidates[0] if hasattr(response, 'candidates') and response.candidates else None
                grounding_metadata = getattr(candidate, 'grounding_metadata', None) if candidate else None
                response_text = getattr(response, 'text', "")
                citations = []
                # add_citations=Trueで引用挿入
                if candidate and grounding_metadata:
                    citation_text, citations = agent._add_citations_to_text(response_text, grounding_metadata)
                result = SearchResult(
                    query=query,
                    response=citation_text,
                    citations=citations,
                    grounding_metadata=agent._extract_grounding_metadata(grounding_metadata) if grounding_metadata else None
                )
                # 会話履歴に追加
                agent.conversation_history.append(types.Content(role="user", parts=[types.Part(text=query)]))
                agent.conversation_history.append(types.Content(role="model", parts=[types.Part(text=response_text)]))
            else:
                # API実行
                result = agent.chat(query, add_citations=False, save_raw_response=(i==0), raw_response_path=response_file)

            print(f"アシスタント: {result.response}")
            print(f"引用数: {len(result.citations)}")
            print(f"会話履歴: {agent.get_conversation_length()}件")

            if result.citations:
                print("引用元:")
                for citation in result.citations:
                    print(f"  {citation['index']}. {citation['title']}")
            
            # 最初の質問のみ詳細表示
            if i == 0 and result.grounding_metadata:
                print(f"\n=== メタデータ ===")
                print(f"検索クエリ: {result.grounding_metadata.get('web_search_queries', [])}")
                print(f"Web結果数: {len(result.grounding_metadata.get('web_results', []))}")
        
    except Exception as e:
        print(f"テスト実行エラー: {str(e)}")
        print("環境変数GEMINI_API_KEYが設定されているか確認してください")