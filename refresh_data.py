import time
import datetime as dt

from parse_data.parse_config import dir_to_save, channel_list, parse_from_date
from parse_data.parsing_collect import parse_and_write_data
from aggregation.predict import predict_pipeline
from utils import read_config, count_start_date_from_end_date


def start_parsing(last_parse_date):
    parse_and_write_data(dir_to_save, 'parsed_data.csv', last_parse_date=last_parse_date, channel_list=channel_list,
                         create_new_csv_flag=True)
    params = read_config()

    today_date = dt.datetime.now().strftime(params["date_format"])
    day_date = count_start_date_from_end_date(today_date, "day")
    week_date = count_start_date_from_end_date(today_date, "week")
    month_date = count_start_date_from_end_date(today_date, "month")
    year_date = count_start_date_from_end_date(today_date, "year")

    predict_pipeline(day_date, today_date, params["paths"]["path_to_predictions"])
    predict_pipeline(week_date, today_date, params["paths"]["path_to_predictions"])
    predict_pipeline(month_date, today_date, params["paths"]["path_to_predictions"])
    predict_pipeline(year_date, today_date, params["paths"]["path_to_predictions"])

    print(f"Parse from {last_parse_date} finished")
    return "Parsing done"


def refresh_data():
    start_parsing(parse_from_date)
    while True:
        time.sleep(86400 // 3)
        params = read_config()
        today_date = dt.datetime.now().strftime(params["date_format"])
        day_date = count_start_date_from_end_date(today_date, "day")

        start_parsing(day_date)
        print(f"Now parse date: {day_date}")

