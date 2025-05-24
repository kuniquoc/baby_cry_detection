from datetime import datetime, timezone, timedelta

def convert_utc_timestamp_to_vn_datetime(timestamp: float) -> datetime:
    """
    Chuyển đổi timestamp UTC thành datetime theo múi giờ Việt Nam (UTC+7).
    
    :param timestamp: Unix timestamp (tính theo UTC)
    :return: datetime theo múi giờ Việt Nam (UTC+7)
    """
    vn_tz = timezone(timedelta(hours=7))
    vn_datetime = datetime.fromtimestamp(timestamp, tz=vn_tz)
    return vn_datetime