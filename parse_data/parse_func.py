from __future__ import annotations
import nest_asyncio
from telethon import TelegramClient
import telethon.sync
import asyncio
from datetime import datetime
from parse_data.parse_config import API_ID, API_HASH


async def get_channel_info(channel_name: str, api_id: int, api_hash: int,
                           parsed_last_time: datetime = None, limit: int = None):
    async with TelegramClient('parse_data/session', api_id, api_hash) as client:
        channel = await client.get_entity(channel_name)
        # date = pd.to_datetime(parsed_last_time, format = date_format)
        reverse = True
        if not parsed_last_time:
            reverse = False
        # offset_date: Дата смещения (будут извлечены сообщения, предшествующие этой дате).
        #reverse: If set to True, the messages will be returned in reverse order (from oldest to newest, instead of the default newest to oldest).
        messages = await client.get_messages(channel, offset_date=parsed_last_time, limit=limit, reverse=reverse,
                                             wait_time=2)
        return messages
