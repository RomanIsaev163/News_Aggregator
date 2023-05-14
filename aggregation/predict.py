from get_response import get_response
from read_write_funcs import format_json_response, save_prediction


def predict_pipeline(start_date: str, end_date: str, path_to_save: str):
    digest, trends, insights = get_response(start_date, end_date)

    response = format_json_response(digest, trends, insights, start_date, end_date)

    save_prediction(response, start_date, end_date, path_to_save=path_to_save)

    return