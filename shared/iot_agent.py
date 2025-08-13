import json
import logging
from .switchbot import control_light, control_aircon, get_device_list, get_device_status
from langchain_core.messages import ToolMessage

# IoT Agent Core Functions
def initialize_devices(token, secret):
    """デバイス情報を初期化する"""
    device_list = get_device_list(token, secret)
    
    # Get light and aircon device IDs
    light_device_id = next((device['deviceId'] for device in device_list['body']['infraredRemoteList'] 
                           if device['remoteType'] == 'Light'), None)
    
    aircon_device_id = next((device['deviceId'] for device in device_list['body']['infraredRemoteList'] 
                            if device['remoteType'] == 'Air Conditioner'), None)
    
    # Find Hub 2 device
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

def light_control(device_id, token, secret, power_state):
    """ライトを制御する"""
    if not device_id:
        return {"error": "ライトデバイスが見つかりません"}
    return control_light(device_id, token, secret, power_state)

def aircon_control(device_id, token, secret, power_state, temp=24, mode=2, fan_speed=3):
    """エアコンを制御する"""
    if not device_id:
        return {"error": "エアコンデバイスが見つかりません"}
    return control_aircon(device_id, token, secret, temp, mode, fan_speed, power_state)

def get_room_environment(device_id, token, secret):
    """室内の温度、湿度、照度を取得する"""
    if not device_id:
        return {
            "error": "Hub 2デバイスが見つかりません。室内環境情報を取得できません。"
        }
    
    # Hub 2から環境情報を取得
    hub_status = get_device_status(device_id, token, secret)
    
    if hub_status['statusCode'] != 100:
        return {
            "error": f"環境情報の取得に失敗しました: {hub_status.get('message', '不明なエラー')}"
        }
    
    # 環境データを抽出
    status_data = hub_status['body']
    environment_info = {
        "temperature": status_data.get('temperature', None),
        "humidity": status_data.get('humidity', None),
        "lightLevel": status_data.get('lightLevel', None),
    }
    
    return environment_info

def process_tool_calls(tool_calls, device_ids, token, secret):
    """ツール呼び出しを処理し、ツールメッセージを生成する
    
    Args:
        tool_calls (list): OpenAIツール呼び出しのリスト
        device_ids (dict): デバイスID情報を含む辞書
        token (str): SwitchBotトークン
        secret (str): SwitchBotシークレット
        
    Returns:
        list: ToolMessageのリスト
    """
    if not tool_calls:
        raise ValueError("ツールコールが空です。有効なツールコールが必要です。")
    
    light_device_id = device_ids.get('light_device_id')
    aircon_device_id = device_ids.get('aircon_device_id')
    hub2_device_id = device_ids.get('hub2_device_id')
    
    tool_messages = []
    
    for tool_call in tool_calls:
        try:
            function_name = tool_call.get('name')
            if not function_name:
                raise ValueError("関数名が指定されていません")
            
            tool_call_id = tool_call.get('id', 'unknown')
            function_args = tool_call.get('args', {})
                
        except Exception as e:
            raise ValueError(f"ツールコール処理中にエラーが発生しました: {str(e)}")
        
        # 各機能に対応する処理
        if function_name == "light_control":
            power_state = function_args.get("power_state")
            if not power_state:
                raise ValueError("power_state パラメータが必要です")
                
            result = light_control(light_device_id, token, secret, power_state)
            logging.info(f"IoT-Agent: ライト{power_state}制御実行")
            
        elif function_name == "aircon_control":
            power_state = function_args.get("power_state")
            if not power_state:
                raise ValueError("power_state パラメータが必要です")
                
            # デフォルト値はここで設定する
            temp = function_args.get("temp", 24)
            mode = function_args.get("mode", 2)
            fan_speed = function_args.get("fan_speed", 3)
            
            result = aircon_control(aircon_device_id, token, secret, power_state, temp, mode, fan_speed)
            logging.info(f"IoT-Agent: エアコン{power_state}制御実行: 温度={temp}度, モード={mode}, 風量={fan_speed}")
            
        elif function_name == "get_room_environment":
            # 室内環境情報を取得
            result = get_room_environment(hub2_device_id, token, secret)
            
            if "error" in result:
                logging.error(f"室内環境情報取得エラー: {result['error']}")
            else:
                temp = result.get("temperature")
                humidity = result.get("humidity")
                light = result.get("lightLevel")
                logging.info(f"IoT-Agent: 室内環境情報取得: 温度={temp}°C, 湿度={humidity}%, 照度={light}/20")
        else:
            raise ValueError(f"未知の関数: {function_name}")
            
        # ToolMessageを作成
        tool_messages.append(
            ToolMessage(
                content=json.dumps(result),
                tool_call_id=tool_call_id
            )
        )
    
    return tool_messages

# OpenAI APIツール定義
def get_tools():
    """OpenAI APIで使用するツール定義を返す"""
    return [
        {
            "type": "function",
            "function": {
                "name": "light_control",
                "description": "電気を点けたり消したりします",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "power_state": {
                            "type": "string",
                            "enum": ["on", "off"],
                            "description": "電気のオン/オフ状態。onかoffのどちらかを指定してください。"
                        }
                    },
                    "required": ["power_state"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "aircon_control",
                "description": "エアコンを操作します",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "power_state": {
                            "type": "string",
                            "enum": ["on", "off"],
                            "description": "エアコンのオン/オフ状態。onかoffのどちらかを指定してください。"
                        },
                        "temp": {
                            "type": "integer",
                            "description": "設定温度（摂氏、16〜30度）"
                        },
                        "mode": {
                            "type": "integer",
                            "enum": [0, 1, 2, 3, 4, 5],
                            "description": "運転モード: 0/1=自動, 2=冷房, 3=除湿, 4=送風, 5=暖房"
                        },
                        "fan_speed": {
                            "type": "integer",
                            "enum": [1, 2, 3, 4],
                            "description": "風量: 1=自動, 2=弱, 3=中, 4=強"
                        }
                    },
                    "required": ["power_state"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_room_environment",
                "description": "室内の環境情報（温度、湿度、照度）を取得します",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]

# システムメッセージ
def get_system_message():
    """IoTエージェント用のシステムメッセージを返す"""
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