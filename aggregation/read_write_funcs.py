import pandas as pd
import json
import os

def read_data_by_period(path_to_data: str = "../data/parsed/parsed.csv", start_date: str = None,
                        end_date: str = None, df=None) -> pd.DataFrame:
    """
        Выдeление новостей из заданного временного промежутка.
        Вход:
            df - DataFrame с колонками 'content' и 'date'
            start_date_string - строка задающая начало временного периода,
        в формате yyyy-mm-dd
            end_date_string - строка задающая конец временного периода.
        Выход:
            отфильтрованный по времени DataFrame
    """
    if df is None:
        df = pd.read_csv(path_to_data)

    df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    return df.loc[mask, :]


def format_json_response(digest, trends, insights, start_date, end_date) -> json:
    result = {"dates": {"start_date": start_date, "end_date": end_date}, "insights": [], "trends": [], "digest": []}

    for digest_item, digest_id in zip(digest, range(0, len(digest))):
        result["digest"].append({'id': digest_id, "title": digest_item[0][0], "content": digest_item[0][1], "channel": digest_item[0][2]})

    for trend_item, trend_id in zip(trends, range(0, len(trends))):
        try:
            result["trends"].append({'id': trend_id, "keywords": trend_item[0], "content": trend_item[1]})
        except Exception:
            print(f'trend {trend_id} is invalid')

    for insight_item, insight_id in zip(insights, range(0, len(insights))):
        result["insights"].append({"id": insight_id, "content": insight_item})

    return result


def save_prediction(json_response, start_date, end_date, path_to_save):
    with open(os.path.join(path_to_save, f"{start_date}_{end_date}.json"), "w", encoding='utf-8') as fp:
        json.dump(json_response, fp, ensure_ascii=False)

    return