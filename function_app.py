import json
import os
import logging
import asyncio

import azure.functions as func

# Blueprintをインポート
from mcp_switchbot_tools import bp as switchbot_bp

# Azure Functions のワーカープロセスのログレベルを設定
logging.getLogger("httpx").setLevel(logging.WARNING)

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.utils import is_request_type, is_intent_name
from ask_sdk_model.dialog import ElicitSlotDirective
from ask_sdk_model import Intent, IntentConfirmationStatus, Slot, SlotConfirmationStatus

# 共有モジュールからsmart_speaker_agentをインポート
from shared.smart_speaker_agent import create_smart_speaker_agent


# アプリケーションのログ設定
# # 環境変数の読み込み
from dotenv import load_dotenv
load_dotenv()

# SwitchBot設定
token = os.getenv("SW_TOKEN")
secret = os.getenv("SW_SECRET")

# LLMプロバイダーの設定（環境変数で切り替え可能）
llm_provider = os.getenv("LLM_PROVIDER", "azure_openai")  # デフォルトはAzure OpenAI

# スマートスピーカーエージェントを初期化
smart_speaker_agent = create_smart_speaker_agent(llm_provider)

# 会話履歴の初期化
conversation_history = {}

async def chat_with_agent(user_input, session_id):
    """スマートスピーカーエージェントとチャットして応答を取得する"""
    return await smart_speaker_agent.chat(user_input, session_id, conversation_history)

# Alexa Skill Handler setup
sb = SkillBuilder()
intent_name = "questionIntent"
slot_name = "message"
directive = ElicitSlotDirective(
    slot_to_elicit=slot_name,
    updated_intent=Intent(
        name=intent_name,
        confirmation_status=IntentConfirmationStatus.NONE,
        slots={
            slot_name: Slot(
                name=slot_name,
                value="",
                confirmation_status=SlotConfirmationStatus.NONE,
            )
        },
    ),
)
class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        session_id = handler_input.request_envelope.session.session_id
        speech_text = "起動しました。何をお手伝いしましょうか？"
        logging.info(f"AI-Response: {speech_text}")
        # return handler_input.response_builder.speak(speech_text).response
        return (
        handler_input.response_builder.speak(speech_text)
        .ask("なにか質問ですか")
        # 自由入力の初期化
        .add_directive(directive)
        .response
    )

class QuestionIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("questionIntent")(handler_input) 
    def handle(self, handler_input):
        session_id = handler_input.request_envelope.session.session_id
        input_text = handler_input.request_envelope.request.intent.slots["message"].value
        logging.info(f"Alexa-Skill: ユーザー入力を受信: {input_text}")

        if(not input_text):
            return handler_input.response_builder.speak("何か質問はありませんか？").response
        input_text = input_text.replace(" ", "")
        # スマートスピーカーエージェントに問い合わせ
        response_text = asyncio.run(chat_with_agent(input_text, session_id))
        
        # return handler_input.response_builder.speak(response_text).response
        return (
        handler_input.response_builder.speak(response_text)
        # 自由入力の初期化
        .add_directive(directive).response
        )

class FallbackIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        # QuestionIntentHandler のロジックを再利用
        session_id = handler_input.request_envelope.session.session_id
        input_text = handler_input.request_envelope.request.intent.slots
        logging.info(f"Alexa-Skill: フォールバック入力を受信: {input_text}")
        if not input_text:
            speech_text = "何か質問はありませんか？"
            logging.info(f"AI-Response: {speech_text}")
            return handler_input.response_builder.speak(speech_text).response
        input_text = input_text.replace(" ", "")
        response_text = asyncio.run(chat_with_agent(input_text, session_id))
        return (
            handler_input.response_builder.speak(response_text)
            .add_directive(directive)
            .response
        )

class StopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return (is_intent_name("AMAZON.StopIntent")(handler_input) or
                is_intent_name("AMAZON.CancelIntent")(handler_input))

    def handle(self, handler_input):
        speech_text = "終了します。またお会いしましょう。"
        logging.info(f"AI-Response: {speech_text}")
        return handler_input.response_builder.speak(speech_text).set_should_end_session(True).response

class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # セッション終了理由をログに記録
        request = handler_input.request_envelope.request
        logging.info(f"Alexa-Skill: セッション終了: {request.reason}")
        
        # SessionEndedRequestには音声レスポンスを返せないため、空のレスポンスを返す
        return handler_input.response_builder.response

# Register handlersf
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(QuestionIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(StopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())

# # Azure Function App setup
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# ブループリントを登録
app.register_blueprint(switchbot_bp)

@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    
    logging.info('AzFunc: HTTP トリガー関数でリクエストを処理しています')
    try:
        # Alexaからのリクエストを処理
        request_body = req.get_json()
        skill_response = sb.lambda_handler()(request_body, None)
        # skill_response = "test2"
        return func.HttpResponse(
            body=json.dumps(skill_response),
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            body=json.dumps({"error": str(e)}),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )


@app.route(route="warmup")
def warmup(_: func.HttpRequest) -> func.HttpResponse:
    """
    コールドスタート対策用のウォームアップ関数
    定期的に呼び出すことでFunction Appのインスタンスを温かく保つ
    """
    logging.info('AzFunc: ウォームアップ関数が呼び出されました')
    
    try:
        # 基本的なレスポンスを返す
        response_data = {
            "status": "healthy",
            "message": "Function App is warmed up",
            "timestamp": func.DateTime.utcnow().isoformat()
        }
        
        logging.info('AzFunc: ウォームアップ処理完了')
        
        return func.HttpResponse(
            body=json.dumps(response_data),
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
        
    except Exception as e:
        logging.error(f"Warmup function error: {str(e)}")
        return func.HttpResponse(
            body=json.dumps({"error": str(e)}),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )









