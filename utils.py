import yaml
import datetime as dt


def count_start_date_from_end_date(end_date: str, mode: str) -> str:
    params = read_config()

    days = count_days_by_mode(mode)
    start_date = (dt.datetime.fromisoformat(end_date) - dt.timedelta(days=days)).strftime(params["date_format"])

    return start_date


def count_days_by_mode(mode: str) -> int:
    if mode == "day":
        return 1
    if mode == "week":
        return 7
    if mode == "month":
        return 30
    if mode == "year":
        return 365


def read_config(path: str = "./prediction_config.yaml") -> yaml:
    """Read config"""

    with open(path, 'r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)
