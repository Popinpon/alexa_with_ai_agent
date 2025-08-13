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

def get_headers(token, secret):
    """
    Generate authentication headers for SwitchBot API
    
    Args:
        token (str): SwitchBot API token
        secret (str): SwitchBot API secret
        
    Returns:
        dict: Headers required for API authentication
    """
    nonce = uuid.uuid4()
    t = int(round(time.time() * 1000))
    string_to_sign = '{}{}{}'.format(token, t, nonce)
    string_to_sign = bytes(string_to_sign, 'utf-8')
    secret = bytes(secret, 'utf-8')
    sign = base64.b64encode(hmac.new(secret, msg=string_to_sign, digestmod=hashlib.sha256).digest())
    headers = {
        'Authorization': token,
        'Content-Type': 'application/json; charset=utf8',
        't': str(t),
        'sign': str(sign, 'utf-8'),
        'nonce': str(nonce)
    }
    return headers

def get_device_list(token, secret):
    """
    Get list of all SwitchBot devices
    
    Args:
        token (str): SwitchBot API token
        secret (str): SwitchBot API secret
        
    Returns:
        dict: API response with device list
    """
    url = 'https://api.switch-bot.com/v1.1/devices'
    headers = get_headers(token, secret)
    response = requests.get(url, headers=headers)
    return response.json()

def get_device_status(device_id, token, secret):
    """
    Get the current status of any SwitchBot device using the API
    
    Args:
        device_id (str): The device ID
        token (str): SwitchBot API token
        secret (str): SwitchBot API secret
        
    Returns:
        dict: The device status response from the API
    """
    url = f'https://api.switch-bot.com/v1.1/devices/{device_id}/status'
    headers = get_headers(token, secret)
    response = requests.get(url, headers=headers)
    return response.json()

def control_light(device_id, token, secret, power_state):
    """
    Control a light device
    
    Args:
        device_id (str): The device ID
        token (str): SwitchBot API token
        secret (str): SwitchBot API secret
        power_state (str): 'on' or 'off'
        
    Returns:
        dict: API response
    """
    url = f'https://api.switch-bot.com/v1.1/devices/{device_id}/commands'
    headers = get_headers(token, secret)
    payload = {
        'command': 'turnOn' if power_state == 'on' else 'turnOff',
        'parameter': 'default',
        'commandType': 'command'
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

def control_aircon(device_id, token, secret, temp, mode, fan_speed, power_state):
    """
    Control an air conditioner
    
    Args:
        device_id (str): The device ID
        token (str): SwitchBot API token
        secret (str): SwitchBot API secret
        temp (int): Temperature setting
        mode (int): Mode setting
        fan_speed (int): Fan speed setting
        power_state (str): 'on' or 'off'
        
    Returns:
        dict: API response
    """
    url = f'https://api.switch-bot.com/v1.1/devices/{device_id}/commands'
    headers = get_headers(token, secret)
    payload = {
        'command': 'setAll',
        'parameter': f'{temp},{mode},{fan_speed},{power_state}',
        'commandType': 'command'
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()