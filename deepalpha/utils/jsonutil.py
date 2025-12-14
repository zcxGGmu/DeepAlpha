"""JSON工具
对应Go版本的 jsonutil/pretty.go
"""

import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

def pretty_json(data: Union[Dict, List, Any], indent: int = 2) -> str:
    """
    将数据格式化为漂亮的JSON字符串

    Args:
        data: 要格式化的数据
        indent: 缩进空格数

    Returns:
        格式化后的JSON字符串
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        return json.dumps({"error": f"JSON serialization failed: {str(e)}"}, indent=indent)

def compact_json(data: Union[Dict, List, Any]) -> str:
    """
    将数据格式化为紧凑的JSON字符串（无缩进）

    Args:
        data: 要格式化的数据

    Returns:
        紧凑的JSON字符串
    """
    try:
        return json.dumps(data, separators=(',', ':'), ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        return json.dumps({"error": f"JSON serialization failed: {str(e)}"}, separators=(',', ':'))

def json_to_dict(json_str: str) -> Optional[Dict]:
    """
    将JSON字符串转换为字典

    Args:
        json_str: JSON字符串

    Returns:
        字典对象，解析失败返回None
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None

def merge_json(*json_objects: Dict) -> Dict:
    """
    合并多个JSON对象

    Args:
        *json_objects: 要合并的字典对象

    Returns:
        合并后的字典
    """
    result = {}
    for obj in json_objects:
        if isinstance(obj, dict):
            result.update(obj)
    return result

def flatten_json(data: Dict, prefix: str = '', sep: str = '.') -> Dict:
    """
    扁平化嵌套的JSON对象

    Args:
        data: 嵌套的字典
        prefix: 键前缀
        sep: 分隔符

    Returns:
        扁平化的字典
    """
    items = []
    for k, v in data.items():
        new_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_json(data: Dict, sep: str = '.') -> Dict:
    """
    反扁平化JSON对象

    Args:
        data: 扁平化的字典
        sep: 分隔符

    Returns:
        嵌套的字典
    """
    result = {}
    for key, value in data.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result

def validate_json(json_str: str) -> bool:
    """
    验证JSON字符串是否有效

    Args:
        json_str: JSON字符串

    Returns:
        是否有效
    """
    try:
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def extract_json_values(data: Union[Dict, List], key: str) -> List[Any]:
    """
    从嵌套的JSON中提取指定键的所有值

    Args:
        data: JSON数据
        key: 要提取的键

    Returns:
        值列表
    """
    values = []

    if isinstance(data, dict):
        for k, v in data.items():
            if k == key:
                values.append(v)
            elif isinstance(v, (dict, list)):
                values.extend(extract_json_values(v, key))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                values.extend(extract_json_values(item, key))

    return values

def count_json_keys(data: Union[Dict, List]) -> int:
    """
    递归计算JSON中的所有键数量

    Args:
        data: JSON数据

    Returns:
        键的总数
    """
    count = 0

    if isinstance(data, dict):
        count += len(data)
        for value in data.values():
            if isinstance(value, (dict, list)):
                count += count_json_keys(value)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                count += count_json_keys(item)

    return count

def json_diff(obj1: Dict, obj2: Dict) -> Dict:
    """
    比较两个JSON对象的差异

    Args:
        obj1: 第一个对象
        obj2: 第二个对象

    Returns:
        差异字典
    """
    diff = {
        'added': {},
        'removed': {},
        'changed': {},
        'unchanged': {}
    }

    all_keys = set(obj1.keys()) | set(obj2.keys())

    for key in all_keys:
        if key not in obj1:
            diff['added'][key] = obj2[key]
        elif key not in obj2:
            diff['removed'][key] = obj1[key]
        elif obj1[key] != obj2[key]:
            diff['changed'][key] = {
                'old': obj1[key],
                'new': obj2[key]
            }
        else:
            diff['unchanged'][key] = obj1[key]

    return diff

def format_json_size(data: Union[Dict, List]) -> str:
    """
    格式化JSON数据大小

    Args:
        data: JSON数据

    Returns:
        格式化的大小字符串
    """
    json_str = json.dumps(data, default=str)
    size_bytes = len(json_str.encode('utf-8'))

    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"