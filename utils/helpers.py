from datetime import datetime


def get_latest_date(data_sensor):
    if data_sensor:
        latest_date = max(data_sensor, key=lambda x: x['tanggal'])
        return datetime.strptime(latest_date['tanggal'], "%Y-%m-%d %H:%M:%S")
    return None
