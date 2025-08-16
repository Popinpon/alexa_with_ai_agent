import json
import os
import logging
from dotenv import load_dotenv
load_dotenv()

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

# 共有モジュールからiot_agentとswitchbotをインポート
from shared import iot_agent

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# アプリケーションのログ設定
# # 環境変数の読み込み


# Initialize AzureChatOpenAI
llm = AzureChatOpenAI(
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    temperature=0.1,
    streaming=True,
)

# SwitchBot設定
token = os.getenv("SW_TOKEN")
secret = os.getenv("SW_SECRET")
# # デバイス情報の初期化
def initialize_device_ids():
    try:
        device_info = iot_agent.initialize_devices(token, secret)
        return device_info
    except Exception as e:
        logging.error(f"デバイス初期化エラー: {str(e)}")
        return {
            'light_device_id': None,
            'aircon_device_id': None,
            'hub2_device_id': None
        }

# デバイスIDを初期化
device_ids = initialize_device_ids()
light_device_id = device_ids['light_device_id']
aircon_device_id = device_ids['aircon_device_id'] 
hub2_device_id = device_ids['hub2_device_id']

# OpenAI APIツール定義を取得
tools = iot_agent.get_tools()

# システムメッセージを取得
iot_system_message = iot_agent.get_system_message()

# 会話履歴の初期化
conversation_history = {}

def chat_with_agent(user_input, session_id):
    """エージェントとチャットして応答を取得する"""
    # セッションの会話履歴を取得または初期化
    if session_id not in conversation_history:
        conversation_history[session_id] = [SystemMessage(content=iot_system_message)]
    
    messages = conversation_history[session_id]
    
    # ユーザー入力を追加
    messages.append(HumanMessage(content=user_input))
    
    # AIからの応答を取得
    response = llm.invoke(messages, tools=tools, tool_choice="auto")
    
    # 応答をメッセージ履歴に追加
    messages.append(response)
    
    # ツール呼び出しがある場合は処理
    if hasattr(response, 'tool_calls') and response.tool_calls:
        try:
            # iot_agentモジュールのprocess_tool_calls関数を使用
            tool_messages = iot_agent.process_tool_calls(response.tool_calls, device_ids, token, secret)
            
            # ToolMessageを会話履歴に追加
            messages.extend(tool_messages)
            
            # AIに実行結果を知らせる
            final_response = llm.invoke(messages)
            messages.append(final_response)
            
            result_content = final_response.content
            logging.info(f"AI-Response: {result_content}")
            return result_content
        except ValueError as e:
            error_message = f"ツール実行エラー: {str(e)}"
            logging.error(error_message)
            
            # エラーメッセージをToolMessageとして追加
            error_tool_message = ToolMessage(
                content=json.dumps({"error": str(e)}),
                tool_call_id=response.tool_calls[0].id if response.tool_calls else "error"
            )
            messages.append(error_tool_message)
            
            # AIにエラーを知らせる
            error_response = llm.invoke(messages)
            messages.append(error_response)
            
            result_content = error_response.content
            logging.info(f"AI-Response (Error): {result_content}")
            return result_content
    else:
        result_content = response.content
        logging.info(f"AI-Response: {result_content}")
        return result_content

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
        speech_text = "ホームエージェントを起動しました。何をお手伝いしましょうか？"
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
        # IoTエージェントに問い合わせ
        response_text = chat_with_agent(input_text, session_id)
        
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
        response_text = chat_with_agent(input_text, session_id)
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









