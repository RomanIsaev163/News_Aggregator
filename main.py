from threading import Thread
from flask import Flask, request
import datetime as dt

from refresh_data import refresh_data
from find_response import find_response
from utils import count_days_by_mode, read_config

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def wake_up():
    return "I am alive!"

@app.route('/start_parsing', methods=['POST', 'GET'])
def start_parsing():
    thread = Thread(target=refresh_data)
    thread.start()

    return ""

@app.route('/by_dates', methods=['POST', 'GET'])
def get_response_by_dates():
    if request.method == 'GET':
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
    elif request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']
    else:
        start_date = None
        end_date = None

    params = read_config("./configs/prediction_config.yaml")

    return find_response(start_date=start_date, end_date=end_date,
                         path_to_predictions=params["paths"]["path_to_predictions"])


@app.route('/by_period', methods=['POST', 'GET'])
def get_response_by_period():
    if request.method == 'GET':
        period = request.args.get('period')
    elif request.method == 'POST':
        period = request.form['period']
    else:
        period = None

    if period not in ['day', 'week', 'month', 'year']:
        print("Неверный период!")
        return

    params = read_config("./prediction_config.yaml")

    end_date = dt.datetime.now().strftime(params["date_format"])

    days = count_days_by_mode(period)
    start_date = (dt.date.fromisoformat(end_date) - dt.timedelta(days=days)).strftime(params["date_format"])

    print(start_date, end_date)

    return find_response(start_date=start_date, end_date=end_date,
                         path_to_predictions=params["paths"]["path_to_predictions"], period=period)


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)