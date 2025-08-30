import requests
import json
import time
import hashlib
import hmac
import base64
import uuid
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

class SwitchBotClient:
    def __init__(self, token=None, secret=None):
        """
        Initialize SwitchBot client
        
        Args:
            token (str, optional): SwitchBot API token. If None, uses SW_TOKEN env var.
            secret (str, optional): SwitchBot API secret. If None, uses SW_SECRET env var.
        """
        if token is None:
            token = os.getenv("SW_TOKEN")
        if secret is None:
            secret = os.getenv("SW_SECRET")
            
        if not token or not secret:
            raise ValueError("SW_TOKEN and SW_SECRET must be provided or set as environment variables")
            
        self.token = token
        self.secret = secret

    def get_headers(self):
        """
        Generate authentication headers for SwitchBot API
        
        Returns:
            dict: Headers required for API authentication
        """
        nonce = uuid.uuid4()
        t = int(round(time.time() * 1000))
        string_to_sign = '{}{}{}'.format(self.token, t, nonce)
        string_to_sign = bytes(string_to_sign, 'utf-8')
        secret = bytes(self.secret, 'utf-8')
        sign = base64.b64encode(hmac.new(secret, msg=string_to_sign, digestmod=hashlib.sha256).digest())
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json; charset=utf8',
            't': str(t),
            'sign': str(sign, 'utf-8'),
            'nonce': str(nonce)
        }
        return headers

    def get_device_list(self):
        """
        Get list of all SwitchBot devices
        
        Returns:
            dict: API response with device list
        """
        url = 'https://api.switch-bot.com/v1.1/devices'
        headers = self.get_headers()
        response = requests.get(url, headers=headers)
        return response.json()

    def get_device_status(self, device_id):
        """
        Get the current status of any SwitchBot device using the API
        
        Args:
            device_id (str): The device ID
            
        Returns:
            dict: The device status response from the API
        """
        url = f'https://api.switch-bot.com/v1.1/devices/{device_id}/status'
        headers = self.get_headers()
        response = requests.get(url, headers=headers)
        return response.json()

    def control_light(self, device_id, power_state):
        """
        Control a light device
        
        Args:
            device_id (str): The device ID
            power_state (str): 'on' or 'off'
            
        Returns:
            dict: API response
        """
        url = f'https://api.switch-bot.com/v1.1/devices/{device_id}/commands'
        headers = self.get_headers()
        payload = {
            'command': 'turnOn' if power_state == 'on' else 'turnOff',
            'parameter': 'default',
            'commandType': 'command'
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()

    def control_aircon(self, device_id, temp, mode, fan_speed, power_state):
        """
        Control an air conditioner
        
        Args:
            device_id (str): The device ID
            temp (int): Temperature setting
            mode (int): Mode setting
            fan_speed (int): Fan speed setting
            power_state (str): 'on' or 'off'
            
        Returns:
            dict: API response
        """
        url = f'https://api.switch-bot.com/v1.1/devices/{device_id}/commands'
        headers = self.get_headers()
        payload = {
            'command': 'setAll',
            'parameter': f'{temp},{mode},{fan_speed},{power_state}',
            'commandType': 'command'
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()