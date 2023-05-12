from __future__ import annotations
import re
import telethon

from telethon import TelegramClient, functions, types, helpers
from datetime import datetime
from parse_data.parse_config import date_format, dir_to_save
import pandas as pd
import os


# функции по обработке текста
def remove_emoji(text: str) -> str:
    """
    В тексте остаются только буквы рус/англ языка, знаки препинания и \n
    """
    if not text:
        return text
    cleanr = re.compile("[^A-Za-z0-9А-Яа-яёЁ' ' '.' ',' '!' '?' '\n' '-' '$' '%' '_' '\-' '\+' '/']" )
    return re.sub(cleanr, '', text)


def remove_double_space(text: str) -> str:
    return re.sub(r'\s+', ' ', text)


def change_enter_to_space(text: str) -> str:
    return re.sub(r'\n+', ' ', text)


def fix_text(text: str) -> str:
    text = remove_emoji(text)
    text = change_enter_to_space(text)
    text = remove_double_space(text)
    text = text.strip()
    return text


# Функции для поиска нежелательных элементов
def find_MessageEntityMention(news_entities: list[telethon.tl.types]) -> telethon.tl.types.MessageEntityMention or None:
    """
    поиск в списке элементов чеей тип MessageEntityMention
    """
    if not news_entities:
        return None
    mention_entities = []
    for entity in news_entities:
        if type(entity) == types.MessageEntityMention:
            mention_entities.append(entity)
    return mention_entities


def find_MessageEntityUrl(news_entities: list[telethon.tl.types]) -> telethon.tl.types.MessageEntityUrl or None:
    if not news_entities:
        return None
    url_entities = []
    for entity in news_entities:
        if type(entity) == types.MessageEntityUrl:
            url_entities.append(entity)
        return url_entities


def find_MessageEntityTextUrl(msg: str,
                              news_entities: list[telethon.tl.types]) -> telethon.tl.types.MessageEntityTextUrl or None:
    if not news_entities:
        return None
    texturl_entities = []
    for entity in news_entities:
        if (type(entity) == types.MessageEntityTextUrl and
                'http' in msg[entity.offset - 1:entity.offset + entity.length + 1]):
            texturl_entities.append(entity)
        return texturl_entities


def find_MessageEntityBold(news_entities: list[telethon.tl.types]) -> telethon.tl.types.MessageEntityBold or None:
    if not news_entities:
        return None
    bold_entity = None
    for entity in news_entities:
        if type(entity) == types.MessageEntityBold:
            bold_entity = entity
            break
    return bold_entity


def find_MessageEntityHashtag(news_entities: list[telethon.tl.types]) \
        -> telethon.tl.types.MessageEntityHashtag:
    if not news_entities:
        return None
    hashtag_entity = []
    for entity in news_entities:
        if type(entity) == types.MessageEntityHashtag:
            hashtag_entity.append(entity)
    return hashtag_entity


def find_Entity_range(msg: str, news_entities: list[telethon.tl.types]) -> list[tuple[int, int]] or None:

    """
    Формироавание кусков (начало, конец) текста которые вырежем
    """
    entities_to_remove = []
    mention_entities = find_MessageEntityMention(news_entities)
    url_entities = find_MessageEntityUrl(news_entities)
    texturl_entities = find_MessageEntityTextUrl(msg, news_entities)
    hashtag_entities = find_MessageEntityHashtag(news_entities)
    entities_to_remove.append(mention_entities)
    entities_to_remove.append(url_entities)
    entities_to_remove.append(texturl_entities)
    entities_to_remove.append(hashtag_entities)

    if not entities_to_remove:
        return None
    slice_lst = []
    for entity_type in entities_to_remove:
        if not entity_type:
            continue
        for entity in entity_type:
            slice_lst.append((entity.offset, entity.offset + entity.length))
    return slice_lst


# Функции очистки каждого сообщение и формирование dataframe title, content, date %Y-%m-%d
def convert_datetime(date: datetime) -> datetime:
    """ Зануляем часы, минуты и тд, оставляеем только год, месяц, день """
    return pd.to_datetime(date.strftime('%Y-%m-%d'))


def get_pure_msg(msg: str, news_entities: list[telethon.tl.types]) -> str:
    """ Формируем новый текст путем удаления нежелательных элементов"""
    slice_lst = find_Entity_range(msg, news_entities)
    if not slice_lst:
        return msg
    sorted_slice_lst = sorted(slice_lst, key=lambda x: x[0])
    pure_msg = ''
    begin = 0
    for slice_ in sorted_slice_lst:
        end = slice_[0]
        pure_msg += msg[begin:end]
        begin = slice_[1]
    return pure_msg


def get_title_text_from_msg(news_text: str) -> tuple[str, str]:
    """
    Разбиваем текст на заголовок и ссодержание новости. Разбивка по первому переносу строки
    Если таково нет, то весь текст идет в title и content
    """
    find_enter_ind = news_text.find('\n')
    if find_enter_ind == -1:
        title = news_text
    else:
        title = news_text[:find_enter_ind]
    news_text = news_text[find_enter_ind + 1:]
    return title, news_text


def preprocess_data_to_df(output: telethon.helpers.TotalList) -> pd.DataFrame:
    """Чистим каждое сообщение из выхода парсера функциями выше и создаем dataframe"""
    titles, contents, dates, channel_names = [], [], [], []
    r = re.compile(r'\bhttps.+?(?:\s|$)')
    for out_news in output:
        pure_msg = get_pure_msg(out_news.message, out_news.entities)
        if not pure_msg:
            continue
        if 'опрос' in pure_msg:
            continue
        pure_msg = re.sub(r, '', pure_msg)

        news_title, news_text = get_title_text_from_msg(pure_msg)
        news_title = fix_text(news_title)
        news_text = fix_text(news_text)
        if (len(news_text.split()) + len(news_title.split())) < 30:
            continue
        titles.append(news_title)
        if len(news_text) < 2:
            news_text = None
        contents.append(news_text)
        news_date = convert_datetime(out_news.date)
        dates.append(news_date)
        channel_names.append(out_news.chat.username)

    news_dataframe = pd.DataFrame({'title': titles, 'content': contents, 'date': dates, 'channel_name': channel_names})
    if not news_dataframe['date'].empty:
        news_dataframe['date'] = news_dataframe['date'].dt.normalize()
    #news_dataframe = del_equal_title_content(news_dataframe)
    return news_dataframe


# Загрузка в csv
def creat_csv(df: pd.DataFrame, channel_name: str) -> None:
    """
    Сохраняем df в csv без индексовой колонки,
    в заданом формате времени в папку указанную глобально в dir_to_save
    с именем название канала + _parsed.csv
    """
    path_to_save = os.path.join(dir_to_save, channel_name + '_parsed.csv')
    df.to_csv(path_to_save, index=False, date_format=date_format)


def del_dup_na(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляем None и дубликаты"""
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def del_equal_title_content(df: pd.DataFrame) -> pd.DataFrame:
    """
    Не все сообщение удается разбить на title, content
    поэтому содержание колонок совпадает
    Функция полностью убирает строчку с таким дубликатом в колонках
    """
    return df[df['content'] != df['title']]