from parse_data.parse_config import channel_list, API_ID, API_HASH, date_format
from datetime import datetime
import pandas as pd
from asyncio import run
from parse_data.parse_func import get_channel_info
from parse_data.text_preprocess_funcs import preprocess_data_to_df, del_dup_na, creat_csv
import os



def parse_selected_tg_to_df(date: str = None, channel_list: list[str] = channel_list,
                            limit=None, creat_csvs_flag: bool = False) -> pd.DataFrame:
    """
  :param date: Дата с которой начнется парсинг. Если не указана, то парсинг каждого канала начнется с последнего
   сообщения до limit
  :param channel_list: Список тг-каналов для парсинга.
  :param limit: Лимит спаршенных сообщений. Не указывать если date не None
  :param creat_csvs_flag: True: в указанной директории dir_to_save создадудтся csv файлы для КАЖДОГО канала
  :return: DataFrame ['title', 'content', 'date', 'channel_name'] с данными по всем каналам с даты date->now
  """
    full_df = pd.DataFrame(columns=['title', 'content', 'date', 'channel_name'])
    date = pd.to_datetime(date)
    for channel_name in channel_list:
        print(f'start parsing {channel_name}', end='\n')
        out = run(get_channel_info(channel_name, API_ID, API_HASH, parsed_last_time=date, limit=limit))
        print(f'end parsing {channel_name}', end='\n')
        print(f'start preprocess {channel_name} data', end='\n')
        df = preprocess_data_to_df(out)
        df = del_dup_na(df)
        if creat_csvs_flag:
            creat_csv(df, channel_name)
        print(f'end preprocess {channel_name} data', end='\n')
        full_df = pd.concat([full_df, df], ignore_index=True)
    print(f'end parsing', end='\n')
    full_df.sort_values(by='date', inplace=True, ignore_index=True)
    return full_df


def drop_duplicates_na(df):
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['title'], inplace=True)
    df.drop_duplicates(subset=['content'], inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def create_parsed_data_file(path_to_parsed_data_file_dir: str, new_df: pd.DataFrame,
                            file_name_csv: str) -> None:
    print('создаю новый файл')
    parsed_data_file_path = os.path.join(path_to_parsed_data_file_dir, file_name_csv)
    new_df = drop_duplicates_na(new_df)
    new_df.to_csv(parsed_data_file_path, index=False, date_format=date_format)


def update_parsed_data_file(path_to_parsed_data_file_dir: str, new_df: pd.DataFrame,
                            file_name_csv: str, create_new_flag: bool = False) -> None:

    parsed_data_file_path = os.path.join(os.path.abspath(path_to_parsed_data_file_dir), file_name_csv)
    try:
        df_to_upd = pd.read_csv(parsed_data_file_path)
        merged_df = pd.concat([df_to_upd, new_df], ignore_index=True)
        merged_df = drop_duplicates_na(merged_df)
        merged_df.to_csv(parsed_data_file_path, index=False, date_format=date_format)
        print('Допарсил')
    except:
        create_parsed_data_file(path_to_parsed_data_file_dir, new_df, file_name_csv)





def parse_and_write_data(path_to_parsed_data_file_dir: str, file_name_csv: str, create_new_csv_flag: bool = False,
                         last_parse_date: str = None, channel_list: list[str] = channel_list, limit=None) -> None:
    """
  Основная функция для парсинга, обработки и записи/обновления.
  :param path_to_parsed_data_file_dir: Директория в которой создастся/обновится csv-файл с спаршенными данными
  :param file_name_csv: Имя файла со всеми спаршенными данными.
  :param create_new_csv_flag: True: создастся file_name_csv, False: обновится file_name_csv
  :param last_parse_date: С этой даты начнется парсинг всех тг-каналов. Если не указана, то запарсится limit сообщений
  начиная с текущей даты
  :param channel_list: Список тг-каналов для парсинга
  :param limit: Лимит спаршенных сообщений. Не указывать если date не None
  :return:
  """
    new_df = parse_selected_tg_to_df(last_parse_date, channel_list, limit=limit)
    update_parsed_data_file(path_to_parsed_data_file_dir, new_df, file_name_csv, create_new_flag=create_new_csv_flag)