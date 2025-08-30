import azure.functions as func
import json
import logging
from dotenv import load_dotenv
from shared.switchbot import SwitchBotClient

# 環境変数を読み込み
load_dotenv()

# Blueprintを作成
bp = func.Blueprint()

# SwitchBotクライアントを初期化
switchbot_client = SwitchBotClient()

# ツールプロパティの定義
tool_properties_get_devices_json = json.dumps([])

tool_properties_get_device_status_json = json.dumps([
    {
        "propertyName": "device_id",
        "propertyType": "string",
        "description": "状態を取得するデバイスのID"
    }
])

tool_properties_control_light_json = json.dumps([
    {
        "propertyName": "device_id",
        "propertyType": "string",
        "description": "ライトデバイスのID"
    },
    {
        "propertyName": "power_state",
        "propertyType": "string",
        "description": "電気のオン/オフ状態。onかoffのどちらかを指定してください。"
    }
])

tool_properties_control_aircon_json = json.dumps([
    {
        "propertyName": "device_id",
        "propertyType": "string",
        "description": "エアコンデバイスのID"
    },
    {
        "propertyName": "temperature",
        "propertyType": "integer",
        "description": "設定温度（摂氏、16〜30度）"
    },
    {
        "propertyName": "mode",
        "propertyType": "integer",
        "description": "運転モード: 1=自動, 2=冷房, 3=除湿, 4=送風, 5=暖房"
    },
    {
        "propertyName": "fan_speed",
        "propertyType": "integer",
        "description": "風量: 1=自動, 2=弱, 3=中, 4=強"
    },
    {
        "propertyName": "power_state",
        "propertyType": "string",
        "description": "エアコンのオン/オフ状態。onかoffのどちらかを指定してください。"
    }
])

@bp.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="get_switchbot_devices", 
    description="SwitchBotデバイスの一覧を取得します。接続されているデバイスとリモコンデバイスの情報を返します",
    toolProperties=tool_properties_get_devices_json,
)
def get_switchbot_devices(context) -> str:
    """Get list of all SwitchBot devices"""
    try:
        result = switchbot_client.get_device_list()
        logging.info(f"Retrieved device list: {len(result.get('body', {}).get('deviceList', []))} devices")
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logging.error(f"Error getting device list: {str(e)}")
        return json.dumps({"error": str(e)})

@bp.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="get_device_status",
    description="指定したSwitchBotデバイスの現在の状態（温度、湿度、照度など）を取得します",
    toolProperties=tool_properties_get_device_status_json,
)
def get_switchbot_device_status(context) -> str:
    """Get device status by device ID"""
    try:
        content = json.loads(context)
        device_id = content["arguments"]["device_id"]
        
        if not device_id:
            return json.dumps({"error": "device_id is required"})
        
        result = switchbot_client.get_device_status(device_id)
        logging.info(f"Retrieved status for device {device_id}")
        logging.info(f"result: {result}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logging.error(f"Error getting device status: {str(e)}")
        return json.dumps({"error": str(e)})

@bp.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="control_light",
    description="電気を点けたり消したりします。SwitchBotの赤外線リモコンでライトを制御します",
    toolProperties=tool_properties_control_light_json,
)
def control_switchbot_light(context) -> str:
    """Control SwitchBot light device"""
    try:
        content = json.loads(context)
        device_id = content["arguments"]["device_id"]
        power_state = content["arguments"]["power_state"]
        
        if not device_id or not power_state:
            return json.dumps({"error": "device_id and power_state are required"})
        
        if power_state not in ["on", "off"]:
            return json.dumps({"error": "power_state must be 'on' or 'off'"})
        
        result = switchbot_client.control_light(device_id, power_state)
        logging.info(f"Controlled light {device_id} - {power_state}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logging.error(f"Error controlling light: {str(e)}")
        return json.dumps({"error": str(e)})

@bp.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="control_aircon",
    description="エアコンを操作します。温度、運転モード、風量を設定してSwitchBotの赤外線リモコンでエアコンを制御します",
    toolProperties=tool_properties_control_aircon_json,
)
def control_switchbot_aircon(context) -> str:
    """Control SwitchBot air conditioner"""
    try:
        content = json.loads(context)
        args = content["arguments"]
        
        device_id = args["device_id"]
        temperature = args["temperature"] 
        mode = args["mode"]
        fan_speed = args["fan_speed"]
        power_state = args["power_state"]
        
        # バリデーション
        if not (16 <= temperature <= 30):
            return json.dumps({"error": "Temperature must be between 16-30°C"})
        if not (1 <= mode <= 5):
            return json.dumps({"error": "Mode must be between 1-5"})
        if not (1 <= fan_speed <= 4):
            return json.dumps({"error": "Fan speed must be between 1-4"})
        if power_state not in ["on", "off"]:
            return json.dumps({"error": "power_state must be 'on' or 'off'"})
        
        result = switchbot_client.control_aircon(
            device_id, temperature, mode, fan_speed, power_state
        )
        logging.info(f"Controlled aircon {device_id} - temp:{temperature}, mode:{mode}, fan:{fan_speed}, power:{power_state}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logging.error(f"Error controlling aircon: {str(e)}")
        return json.dumps({"error": str(e)})