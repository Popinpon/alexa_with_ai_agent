"""
SwitchBot デバイス管理クラス
"""
import logging
import time
from typing import Dict, Any, List
from langchain_core.tools import tool
from shared.switchbot import SwitchBotClient

logger = logging.getLogger(__name__)


class SwitchBotManager:
    """SwitchBot デバイスの管理とツール作成を行うクラス"""
    
    def __init__(self):
        self.switchbot_client = SwitchBotClient()
        # デバイス情報キャッシュ
        self._device_cache = None
        self._device_cache_timestamp = None
        self._cache_ttl = 300  # 5分間有効
    
    def _is_cache_valid(self) -> bool:
        """キャッシュが有効かどうかを確認"""
        if self._device_cache is None or self._device_cache_timestamp is None:
            return False
        
        current_time = time.time()
        return (current_time - self._device_cache_timestamp) < self._cache_ttl
    
    def _get_switchbot_devices(self) -> Dict[str, Any]:
        """SwitchBotクライアントを使用してデバイス情報を取得（キャッシュ機能付き）"""
        # キャッシュが有効な場合はキャッシュから返す
        if self._is_cache_valid():
            logger.info("📋 Using cached device information")
            return self._device_cache
        
        try:
            result = self.switchbot_client.get_device_list()
            
            # レスポンスの'body'部分を取得
            if 'body' in result:
                device_data = result['body']
            else:
                device_data = result
            
            # キャッシュに保存
            if device_data:
                self._device_cache = device_data
                self._device_cache_timestamp = time.time()
                logger.info(f"💾 Device info cached")
            
            return device_data if device_data else {}
            
        except Exception as e:
            logger.error(f"SwitchBotデバイス取得エラー: {e}")
            return {}
    
    def get_actual_device_ids(self) -> Dict[str, str]:
        """実際のデバイス情報を取得（IoT操作時に使用）"""
        # SwitchBotデバイス一覧を取得（キャッシュ機能付き）
        devices_info = self._get_switchbot_devices()
        
        if not devices_info:
            logger.warning("SwitchBotデバイスが取得できませんでした。")

        device_mapping = {}
        
        # iot_agent.pyの実装に合わせてデバイスマッピング
        # Get light and aircon device IDs from infraredRemoteList
        light_device_id = next((device['deviceId'] for device in devices_info.get('infraredRemoteList', [])
                                if device['remoteType'] == 'Light'), None)
        
        aircon_device_id = next((device['deviceId'] for device in devices_info.get('infraredRemoteList', [])
                                if device['remoteType'] == 'Air Conditioner'), None)
        
        # Find Hub 2 device from deviceList
        hub2_device_id = None
        for device in devices_info.get('deviceList', []):
            if device.get('deviceType') == 'Hub 2':
                hub2_device_id = device['deviceId']
                logger.info(f"SmartSpeaker-Agent: Hub 2デバイスを検出: {device.get('deviceName', 'Unknown')} (ID: {hub2_device_id})")
                break
        
        if not hub2_device_id:
            logger.warning("警告: Hub 2デバイスが見つかりません。室内環境情報の取得ができません。")
        
        device_mapping = {
            'light_device_id': light_device_id,
            'aircon_device_id': aircon_device_id,
            'hub2_device_id': hub2_device_id
        }
        
        logger.info(f"SwitchBotデバイスマッピング: {device_mapping}")
        return device_mapping
    
    def get_default_devices(self) -> Dict[str, str]:
        """デフォルトのデバイス設定を返す"""
        return {
            'light_device_id': "02-202403301114-45200468",
            'aircon_device_id': "02-202504191706-42866040", 
            'hub2_device_id': "C6FD9F3D1826"
        }
    
    def create_switchbot_tools(self) -> List:
        """SwitchBotツールを作成"""
        switchbot_tools_start = time.time()
        
        @tool
        def get_switchbot_devices() -> Dict[str, Any]:
            """SwitchBotデバイス一覧を取得します"""
            try:
                return self.switchbot_client.get_device_list()
            except Exception as e:
                logger.error(f"SwitchBotデバイス取得エラー: {e}")
                return {"error": str(e)}
        
        @tool
        def get_device_status(device_id: str) -> Dict[str, Any]:
            """指定したSwitchBotデバイスの状態を取得します
            
            Args:
                device_id: デバイスID
            """
            try:
                return self.switchbot_client.get_device_status(device_id)
            except Exception as e:
                logger.error(f"デバイス状態取得エラー: {e}")
                return {"error": str(e)}
        
        @tool
        def control_light(device_id: str, power_state: str) -> Dict[str, Any]:
            """ライトを制御します
            
            Args:
                device_id: デバイスID
                power_state: 'on' または 'off'
            """
            try:
                return self.switchbot_client.control_light(device_id, power_state)
            except Exception as e:
                logger.error(f"ライト制御エラー: {e}")
                return {"error": str(e)}
        
        @tool
        def control_aircon(device_id: str, temperature: int, mode: int, fan_speed: int, power_state: str) -> Dict[str, Any]:
            """エアコンを制御します
            
            Args:
                device_id: デバイスID
                temperature: 温度設定（16-30）
                mode: モード（1:自動, 2:冷房, 3:除湿, 4:送風, 5:暖房）
                fan_speed: 風量（1:自動, 2:弱, 3:中, 4:強）
                power_state: 'on' または 'off'
            """
            try:
                return self.switchbot_client.control_aircon(device_id, temperature, mode, fan_speed, power_state)
            except Exception as e:
                logger.error(f"エアコン制御エラー: {e}")
                return {"error": str(e)}
        
        tools = [get_switchbot_devices, get_device_status, control_light, control_aircon]
        switchbot_tools_time = time.time() - switchbot_tools_start
        logger.info(f"⏱️ SwitchBot tools creation: {switchbot_tools_time:.3f}s")
        
        return tools