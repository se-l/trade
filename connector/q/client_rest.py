import json
import requests

from typing import Union, List

home_kdb = r'/repos/trade/data/kdb/'
url_api = "http://localhost:5042"


def path_symbol(tbl: str) -> str: return f':{home_kdb + tbl}'


def create(tbl: str, data: List[dict]) -> bool:
    ps = path_symbol(tbl)
    data = '''([] c1:`a`b`c; c2:10 20 30)'''
    res = requests.post(url_api, data=json.dumps({
        'qsql': f'`{ps} set {data}',
    }))
    return json.loads(res.content) == ps


def append(tbl, data= ''' 9 9 9'''):
    ps = path_symbol(tbl)
    res = requests.post(url_api, data=json.dumps({
        'qsql': f'`.[`{ps}; (); ,; {data}]',
    })).content == ps


def query(sql: str): return get(sql)
def read(sql): pass
def update(): pass
def delete(): pass


def drop(tbl: str) -> bool:
    ps = path_symbol(tbl)
    return post(q=f'hdel `{ps}') == ps


def post(q: str) -> Union[List[dict], None]:
    ret = requests.post(url_api, data=json.dumps({
        'qsql': q
    }))
    if ret.ok:
        return json.loads(ret.content)


def get(sql: str) -> Union[List[dict], None]:
    """Not very practical"""
    ret = requests.get(url_api, params=sql)
    if ret.ok:
        return json.loads(ret.content)


# s='''select distinct p,s.city from sp'''
if __name__ == '__main__':
    sql = '''select distinct p,s.city from sp'''
    sql = '''`:/data/t set ([] c1:`a`b`c; c2:10 20 30)'''
    print('POST')
    print(create('t', None))
    print(drop('t'))
    print(post('''hdel `:/data/t'''))
    # print('GET')
    # print(get(sql))
