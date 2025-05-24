import logging
from typing import Dict, Any
import asyncio
import threading

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from websocket.connection_manager import ConnectionManager
from firebase_service import send_cry_notification, send_nocry_notification

from utils.date_utils import convert_utc_timestamp_to_vn_datetime

logger = logging.getLogger(__name__)

class CryDetectionService:
    def __init__(self, connection_manager: ConnectionManager, no_cry_check_seconds: int = 10):
        self.manager = connection_manager
        self.no_cry_check_seconds = no_cry_check_seconds
        self.cry_status_tracker: Dict[str, Dict[str, Any]] = {}

    def init_device_tracking(self, client_id: str):
        """Initialize or reset cry status tracking for a device"""
        self.cry_status_tracker[client_id] = {
            'last_cry_time': None,
            'checking_no_cry': False,
            'no_cry_timer': None,
            'timer_start_time': None
        }

    def cleanup_device_tracking(self, client_id: str):
        """Clean up tracking resources for a device"""
        if client_id in self.cry_status_tracker:
            if self.cry_status_tracker[client_id]['no_cry_timer'] is not None:
                try:
                    self.cry_status_tracker[client_id]['no_cry_timer'].cancel()
                except:
                    pass
            del self.cry_status_tracker[client_id]

    async def process_cry_detection(self, client_id: str, timestamp: float, confidence: float) -> bool:
        """Process cry detection event and handle notifications"""
        if client_id not in self.cry_status_tracker:
            self.init_device_tracking(client_id)

        device_status = self.cry_status_tracker[client_id]
        
        # Cập nhật thời gian khóc cuối cùng
        device_status['last_cry_time'] = timestamp

        # Nếu đang có timer chạy, hủy nó và tạo timer mới
        if device_status['checking_no_cry']:
            if device_status['no_cry_timer'] and device_status['no_cry_timer'].is_alive():
                device_status['no_cry_timer'].cancel()
                logger.info(f"Reset no-cry timer for device {client_id} - new crying detected")
        
        # Thiết lập timer mới
        device_status['checking_no_cry'] = True
        device_status['timer_start_time'] = timestamp
        
        # Tạo và khởi động timer mới
        no_cry_timer = threading.Timer(
            self.no_cry_check_seconds,
            self._schedule_no_cry_check,
            args=[client_id, timestamp]
        )
        no_cry_timer.daemon = True
        no_cry_timer.start()
        device_status['no_cry_timer'] = no_cry_timer

        # Gửi thông báo crying đến Firebase 
        notification_sent = await send_cry_notification({
            'timestamp': timestamp,
            'deviceId': client_id,
            'confidence': confidence
        })

        return notification_sent

    def _schedule_no_cry_check(self, client_id: str, timer_start_timestamp: float):
        """Schedule the no-cry check to run in the event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._check_for_no_cry(client_id, timer_start_timestamp))
        finally:
            loop.close()

    async def _check_for_no_cry(self, client_id: str, timer_start_timestamp: float):
        """Check if crying has stopped after the timeout period"""
        try:
            if client_id not in self.cry_status_tracker:
                logger.warning(f"Device {client_id} not in tracker when checking for no-cry")
                return

            device_status = self.cry_status_tracker[client_id]

            # If this timer was reset by a new cry detection, do nothing
            if device_status['timer_start_time'] != timer_start_timestamp:
                logger.info(f"No-cry check skipped for device {client_id} - timer was reset")
                return

            # If we get here, the timeout period has passed with no new cry detection
            no_cry_timestamp = timer_start_timestamp + self.no_cry_check_seconds

            # Send no-cry notification
            notification_sent = await send_nocry_notification({
                'timestamp': no_cry_timestamp,
                'deviceId': client_id,
                'lastCryTimestamp': device_status['last_cry_time']
            })

            if notification_sent:
                await self.manager.send_message(client_id, {
                    "type": "alert",
                    "timestamp": no_cry_timestamp,
                    "message": "No crying detected",
                    "deviceId": client_id
                })
                logger.info(
                    f"No-cry event detected at {convert_utc_timestamp_to_vn_datetime(no_cry_timestamp)} for device {client_id}. "
                    f"Last cry was at {device_status['last_cry_time']}, "
                    f"{no_cry_timestamp - device_status['last_cry_time']:.2f}s ago"
                )

            # Reset the timer state
            device_status['checking_no_cry'] = False
            device_status['no_cry_timer'] = None
            device_status['timer_start_time'] = None

        except Exception as e:
            logger.error(f"Error in check_for_no_cry: {str(e)}")