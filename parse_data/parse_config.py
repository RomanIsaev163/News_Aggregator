from __future__ import annotations
import nest_asyncio
import asyncio
from telethon import TelegramClient, functions, types, helpers
from asyncio import run
import telethon
import pandas as pd
from datetime import datetime
import re
import os
import telethon.sync

API_ID = 27950605
API_HASH = '4ec1f36ae431d29e7ecd57ba2853d9c7'
#phone_number = #YOUR PHONE_NUBER
#token = #YOUR TOKEN
channel_list = ['businesstodayy',
                'exploitex', 'ostorozhno_novosti', 'techno_news_tg',
                'd_code', 'FatCat18', 'bolecon', 'AK47pfl']
#channel_list = ['startupoftheday','mytar_rf', 'businesstodayy',
#                'exploitex', 'ostorozhno_novosti', 'techno_news_tg',
#               'd_code', 'FatCat18', 'bolecon', 'AK47pfl']
dir_to_save = "./data/parsed"
date_format = '%Y-%m-%d'
limit = 1000
parse_from_date = '2023-05-10'