#encoding=utf8
import json


def dict_to_json():
    dict = {}
    dict['name'] = 'many'
    dict['age'] = 10
    dict['sex'] = 'male'
    print(dict)  # 输出：{'name': 'many', 'age': 10, 'sex': 'male'}
    j = json.dumps(dict)
    print(j)  # 输出：{"name": "many", "age": 10, "sex": "male"}


if __name__ == '__main__':
    dict_to_json()