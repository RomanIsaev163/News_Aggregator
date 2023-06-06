from parse_data.parsing_collect import parse_and_write_data
from parse_data.parse_config import dir_to_save, parse_from_date, channel_list
parse_and_write_data(dir_to_save, 'parsed.csv', last_parse_date='2023-05-10', channel_list=channel_list)