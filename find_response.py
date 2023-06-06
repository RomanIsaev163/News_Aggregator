import os
import json


def find_response(start_date: str, end_date: str, path_to_predictions: str, period = 'day'):
    for response_name in os.listdir(path_to_predictions):
        dates = response_name[:-5].split('_')
        if dates[0] == start_date and dates[1] == end_date:
            with open(os.path.join(path_to_predictions, response_name), encoding='utf-8') as fp:
                predicted_response = json.load(fp)

            return predicted_response

    with open(os.path.join(path_to_predictions, f"default_{period}.json"), encoding='utf-8') as fp:
        return json.load(fp)