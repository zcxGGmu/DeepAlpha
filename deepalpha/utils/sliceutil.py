"""切片/列表工具
对应Go版本的 sliceutil/
"""

from typing import List, TypeVar, Generic, Callable, Optional, Any, Dict
from copy import deepcopy
import random

T = TypeVar('T')


def clone_slice(items: List[T]) -> List[T]:
    """
    克隆列表（深拷贝）

    Args:
        items: 原始列表

    Returns:
        克隆后的列表
    """
    return deepcopy(items)


def filter_decisions(decisions: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
    """
    过滤决策列表

    Args:
        decisions: 决策列表
        filters: 过滤条件

    Returns:
        过滤后的决策列表
    """
    filtered = []

    for decision in decisions:
        match = True

        for key, value in filters.items():
            if key not in decision:
                match = False
                break

            if isinstance(value, (list, tuple)):
                if decision[key] not in value:
                    match = False
                    break
            elif decision[key] != value:
                match = False
                break

        if match:
            filtered.append(decision)

    return filtered


def group_by(items: List[T], key_func: Callable[[T], Any]) -> Dict[Any, List[T]]:
    """
    按键值分组

    Args:
        items: 列表
        key_func: 键函数

    Returns:
        分组后的字典
    """
    groups = {}
    for item in items:
        key = key_func(item)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups


def unique(items: List[T], key_func: Optional[Callable[[T], Any]] = None) -> List[T]:
    """
    去重

    Args:
        items: 列表
        key_func: 键函数，用于判断唯一性

    Returns:
        去重后的列表
    """
    seen = set()
    result = []

    for item in items:
        key = key_func(item) if key_func else item
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


def chunk(items: List[T], size: int) -> List[List[T]]:
    """
    将列表分块

    Args:
        items: 列表
        size: 块大小

    Returns:
        分块后的列表
    """
    chunks = []
    for i in range(0, len(items), size):
        chunks.append(items[i:i + size])
    return chunks


def flatten(lists: List[List[T]]) -> List[T]:
    """
    扁平化嵌套列表

    Args:
        lists: 嵌套列表

    Returns:
        扁平化后的列表
    """
    return [item for sublist in lists for item in sublist]


def shuffle(items: List[T]) -> List[T]:
    """
    随机打乱列表

    Args:
        items: 列表

    Returns:
        打乱后的列表
    """
    shuffled = items.copy()
    random.shuffle(shuffled)
    return shuffled


def sample(items: List[T], n: int) -> List[T]:
    """
    随机采样

    Args:
        items: 列表
        n: 采样数量

    Returns:
        采样后的列表
    """
    if n >= len(items):
        return items.copy()
    return random.sample(items, n)


def find_index(items: List[T], predicate: Callable[[T], bool]) -> int:
    """
    查找元素索引

    Args:
        items: 列表
        predicate: 谓词函数

    Returns:
        第一个匹配元素的索引，未找到返回-1
    """
    for i, item in enumerate(items):
        if predicate(item):
            return i
    return -1


def find_all(items: List[T], predicate: Callable[[T], bool]) -> List[T]:
    """
    查找所有匹配的元素

    Args:
        items: 列表
        predicate: 谓词函数

    Returns:
        所有匹配的元素
    """
    return [item for item in items if predicate(item)]


def partition(items: List[T], predicate: Callable[[T], bool]) -> tuple[List[T], List[T]]:
    """
    根据谓词分区

    Args:
        items: 列表
        predicate: 谓词函数

    Returns:
        (匹配的元素, 不匹配的元素)
    """
    matched = []
    unmatched = []

    for item in items:
        if predicate(item):
            matched.append(item)
        else:
            unmatched.append(item)

    return matched, unmatched


def sort_by(items: List[T], key_func: Callable[[T], Any], reverse: bool = False) -> List[T]:
    """
    根据键函数排序

    Args:
        items: 列表
        key_func: 键函数
        reverse: 是否逆序

    Returns:
        排序后的列表
    """
    return sorted(items, key=key_func, reverse=reverse)


def take(items: List[T], n: int) -> List[T]:
    """
    取前n个元素

    Args:
        items: 列表
        n: 数量

    Returns:
        前n个元素
    """
    return items[:n]


def drop(items: List[T], n: int) -> List[T]:
    """
    丢弃前n个元素

    Args:
        items: 列表
        n: 数量

    Returns:
        丢弃前n个元素后的列表
    """
    return items[n:]


def take_last(items: List[T], n: int) -> List[T]:
    """
    取最后n个元素

    Args:
        items: 列表
        n: 数量

    Returns:
        最后n个元素
    """
    return items[-n:] if n > 0 else []


def drop_last(items: List[T], n: int) -> List[T]:
    """
    丢弃最后n个元素

    Args:
        items: 列表
        n: 数量

    Returns:
        丢弃最后n个元素后的列表
    """
    return items[:-n] if n > 0 else items[:]


def zip_longest(*lists: List[T]) -> List[tuple]:
    """
    拉链多个列表，长度不一致时填充None

    Args:
        *lists: 多个列表

    Returns:
        元组列表
    """
    from itertools import zip_longest
    return list(zip_longest(*lists))


def interleave(*lists: List[T]) -> List[T]:
    """
    交错合并多个列表

    Args:
        *lists: 多个列表

    Returns:
        交错合并后的列表
    """
    result = []
    min_len = min(len(lst) for lst in lists) if lists else 0

    for i in range(min_len):
        for lst in lists:
            result.append(lst[i])

    # 添加剩余元素
    for lst in lists:
        if len(lst) > min_len:
            result.extend(lst[min_len:])

    return result


def cartesian_product(*lists: List[T]) -> List[tuple]:
    """
    计算笛卡尔积

    Args:
        *lists: 多个列表

    Returns:
        笛卡尔积结果
    """
    from itertools import product
    return list(product(*lists))


def sliding_window(items: List[T], size: int, step: int = 1) -> List[List[T]]:
    """
    滑动窗口

    Args:
        items: 列表
        size: 窗口大小
        step: 步长

    Returns:
        窗口列表
    """
    windows = []
    for i in range(0, len(items) - size + 1, step):
        windows.append(items[i:i + size])
    return windows


def rotate(items: List[T], n: int) -> List[T]:
    """
    旋转列表

    Args:
        items: 列表
        n: 旋转位数（正数向右，负数向左）

    Returns:
        旋转后的列表
    """
    n = n % len(items) if items else 0
    return items[-n:] + items[:-n] if n else items


def bisect_left(items: List[T], value: T, key_func: Optional[Callable[[T], Any]] = None) -> int:
    """
    二分查找插入位置（左）

    Args:
        items: 已排序的列表
        value: 要查找的值
        key_func: 键函数

    Returns:
        插入位置
    """
    import bisect

    if key_func:
        keys = [key_func(item) for item in items]
        return bisect.bisect_left(keys, key_func(value))
    else:
        return bisect.bisect_left(items, value)


def bisect_right(items: List[T], value: T, key_func: Optional[Callable[[T], Any]] = None) -> int:
    """
    二分查找插入位置（右）

    Args:
        items: 已排序的列表
        value: 要查找的值
        key_func: 键函数

    Returns:
        插入位置
    """
    import bisect

    if key_func:
        keys = [key_func(item) for item in items]
        return bisect.bisect_right(keys, key_func(value))
    else:
        return bisect.bisect_right(items, value)


def moving_average(items: List[float], window: int) -> List[float]:
    """
    移动平均

    Args:
        items: 数值列表
        window: 窗口大小

    Returns:
        移动平均值列表
    """
    if window <= 0:
        return []

    result = []
    for i in range(len(items) - window + 1):
        avg = sum(items[i:i + window]) / window
        result.append(avg)

    return result


def accumulate(items: List[T], func: Callable[[T, T], T]) -> List[T]:
    """
    累积计算

    Args:
        items: 列表
        func: 累积函数

    Returns:
        累积结果列表
    """
    from itertools import accumulate
    return list(accumulate(items, func))


def diff(items: List[T]) -> List[T]:
    """
    计算相邻元素的差值

    Args:
        items: 列表

    Returns:
        差值列表
    """
    if len(items) < 2:
        return []

    return [items[i + 1] - items[i] for i in range(len(items) - 1)]