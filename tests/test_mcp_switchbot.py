#!/usr/bin/env python3
"""
Test script for SwitchBot MCP tools
"""

import json
import os
from dotenv import load_dotenv
from mcp_switchbot_tools import (
    get_switchbot_devices,
    get_switchbot_device_status,
    control_switchbot_light,
    control_switchbot_aircon
)

# Load environment variables
load_dotenv()

def test_get_devices():
    """Test getting device list"""
    print("=== Testing get_switchbot_devices ===")
    
    # Simulate context (empty for this tool)
    context = json.dumps({})
    
    result = get_switchbot_devices(context)
    print(f"Result: {result}")
    
    parsed_result = json.loads(result)
    if "error" in parsed_result:
        print(f"Error: {parsed_result['error']}")
    else:
        device_list = parsed_result.get("body", {}).get("deviceList", [])
        print(f"Found {len(device_list)} devices")
        for device in device_list[:3]:  # Show first 3 devices
            print(f"- {device.get('deviceName', 'Unknown')} ({device.get('deviceType', 'Unknown')}) - ID: {device.get('deviceId', 'Unknown')}")
    
    return parsed_result

def test_get_device_status(device_id):
    """Test getting device status"""
    print(f"\n=== Testing get_device_status for {device_id} ===")
    
    context = json.dumps({
        "arguments": {
            "device_id": device_id
        }
    })
    
    result = get_switchbot_device_status(context)
    print(f"Result: {result}")
    
    parsed_result = json.loads(result)
    if "error" in parsed_result:
        print(f"Error: {parsed_result['error']}")
    else:
        body = parsed_result.get("body", {})
        print(f"Device status: {body}")
    
    return parsed_result

def test_control_light(device_id, power_state):
    """Test controlling light"""
    print(f"\n=== Testing control_light for {device_id} - {power_state} ===")
    
    context = json.dumps({
        "arguments": {
            "device_id": device_id,
            "power_state": power_state
        }
    })
    
    result = control_switchbot_light(context)
    print(f"Result: {result}")
    
    parsed_result = json.loads(result)
    if "error" in parsed_result:
        print(f"Error: {parsed_result['error']}")
    else:
        print(f"Light control success: {parsed_result}")
    
    return parsed_result

def test_control_aircon(device_id):
    """Test controlling air conditioner"""
    print(f"\n=== Testing control_aircon for {device_id} ===")
    
    context = json.dumps({
        "arguments": {
            "device_id": device_id,
            "temperature": 24,
            "mode": 2,  # cool
            "fan_speed": 2,  # low
            "power_state": "on"
        }
    })
    
    result = control_switchbot_aircon(context)
    print(f"Result: {result}")
    
    parsed_result = json.loads(result)
    if "error" in parsed_result:
        print(f"Error: {parsed_result['error']}")
    else:
        print(f"Aircon control success: {parsed_result}")
    
    return parsed_result

def main():
    """Main test function"""
    print("SwitchBot MCP Tools Test")
    print("========================")
    
    # Check environment variables
    if not os.getenv("SW_TOKEN") or not os.getenv("SW_SECRET"):
        print("Error: SW_TOKEN and SW_SECRET environment variables are required")
        return
    
    # Test 1: Get devices
    devices_result = test_get_devices()
    
    if "error" not in devices_result:
        device_list = devices_result.get("body", {}).get("deviceList", [])
        infrared_list = devices_result.get("body", {}).get("infraredRemoteList", [])
        
        if device_list:
            # Test with first device
            first_device = device_list[0]
            device_id = first_device.get("deviceId")
            device_type = first_device.get("deviceType", "").lower()
            
            if device_id:
                # Test 2: Get device status
                test_get_device_status(device_id)
                
                # Test 3: Control based on device type
                if "light" in device_type or "bulb" in device_type:
                    test_control_light(device_id, "on")
                elif "airconditioner" in device_type or "aircon" in device_type:
                    test_control_aircon(device_id)
                else:
                    print(f"Device type '{device_type}' not specifically tested")
        
        # Test infrared remote devices
        light_device = None
        aircon_device = None
        
        for device in infrared_list:
            if device.get("remoteType") == "Light":
                light_device = device
            elif device.get("remoteType") == "Air Conditioner":
                aircon_device = device
        
        # Test light control
        if light_device:
            light_id = light_device.get("deviceId")
            test_control_light(light_id, "on")
            test_control_light(light_id, "off")
        
        # Test air conditioner control
        if aircon_device:
            aircon_id = aircon_device.get("deviceId")
            test_control_aircon(aircon_id)
            
        if not device_list and not infrared_list:
            print("No devices found for testing")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    main()