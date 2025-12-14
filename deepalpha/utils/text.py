"""文本工具
对应Go版本的 text/truncate.go
"""

import unicodedata
from typing import List, Optional, Tuple

def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    截断文本到指定长度

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix

def truncate_words(text: str, max_words: int, suffix: str = "...") -> str:
    """
    按单词数截断文本

    Args:
        text: 原始文本
        max_words: 最大单词数
        suffix: 后缀

    Returns:
        截断后的文本
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words]) + suffix

def truncate_middle(text: str, max_length: int, separator: str = "...") -> str:
    """
    从中间截断文本

    Args:
        text: 原始文本
        max_length: 最大长度
        separator: 中间分隔符

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text

    separator_len = len(separator)
    keep_length = (max_length - separator_len) // 2

    return text[:keep_length] + separator + text[-keep_length:]

def ellipsize(text: str, max_length: int) -> str:
    """
    省略号截断，智能处理单词边界

    Args:
        text: 原始文本
        max_length: 最大长度

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text

    # 尝试在单词边界截断
    truncated = text[:max_length - 3]
    last_space = truncated.rfind(' ')

    if last_space > max_length // 2:
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."

def wrap_text(text: str, width: int = 80, indent: str = "") -> str:
    """
    文本换行

    Args:
        text: 原始文本
        width: 行宽
        indent: 缩进

    Returns:
        换行后的文本
    """
    lines = []
    words = text.split()
    current_line = indent

    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            if current_line == indent:
                current_line += word
            else:
                current_line += " " + word
        else:
            lines.append(current_line)
            current_line = indent + word

    if current_line != indent:
        lines.append(current_line)

    return "\n".join(lines)

def remove_accents(text: str) -> str:
    """
    移除文本中的重音符号

    Args:
        text: 原始文本

    Returns:
        无重音符号的文本
    """
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(
        char for char in normalized
        if unicodedata.category(char) != 'Mn'
    )

def slugify(text: str, separator: str = "-") -> str:
    """
    将文本转换为URL友好的slug

    Args:
        text: 原始文本
        separator: 分隔符

    Returns:
        slug字符串
    """
    # 移除重音
    text = remove_accents(text)

    # 转为小写并移除非字母数字字符
    words = []
    for word in text.lower().split():
        cleaned = ''.join(c for c in word if c.isalnum())
        if cleaned:
            words.append(cleaned)

    return separator.join(words)

def capitalize_words(text: str) -> str:
    """
    首字母大写每个单词

    Args:
        text: 原始文本

    Returns:
        首字母大写的文本
    """
    return ' '.join(word.capitalize() for word in text.split())

def camel_case(text: str) -> str:
    """
    转换为驼峰命名

    Args:
        text: 原始文本

    Returns:
        驼峰命名的字符串
    """
    words = text.split()
    if not words:
        return ""

    first = words[0].lower()
    rest = ''.join(word.capitalize() for word in words[1:])

    return first + rest

def snake_case(text: str) -> str:
    """
    转换为蛇形命名

    Args:
        text: 原始文本

    Returns:
        蛇形命名的字符串
    """
    # 处理驼峰命名
    result = []
    for i, char in enumerate(text):
        if char.isupper() and i > 0:
            result.append('_')
            result.append(char.lower())
        else:
            result.append(char.lower() if char != ' ' else '_')

    return ''.join(result).replace('__', '_').strip('_')

def pad_left(text: str, length: int, fill_char: str = " ") -> str:
    """
    左侧填充

    Args:
        text: 原始文本
        length: 目标长度
        fill_char: 填充字符

    Returns:
        填充后的文本
    """
    return text.rjust(length, fill_char)

def pad_right(text: str, length: int, fill_char: str = " ") -> str:
    """
    右侧填充

    Args:
        text: 原始文本
        length: 目标长度
        fill_char: 填充字符

    Returns:
        填充后的文本
    """
    return text.ljust(length, fill_char)

def center_text(text: str, length: int, fill_char: str = " ") -> str:
    """
    居中文本

    Args:
        text: 原始文本
        length: 目标长度
        fill_char: 填充字符

    Returns:
        居中的文本
    """
    return text.center(length, fill_char)

def count_words(text: str) -> int:
    """
    计算单词数

    Args:
        text: 文本

    Returns:
        单词数
    """
    return len(text.split())

def count_chars(text: str, ignore_spaces: bool = False) -> int:
    """
    计算字符数

    Args:
        text: 文本
        ignore_spaces: 是否忽略空格

    Returns:
        字符数
    """
    if ignore_spaces:
        return len(text.replace(" ", ""))
    return len(text)

def reverse_text(text: str) -> str:
    """
    反转文本

    Args:
        text: 原始文本

    Returns:
        反转后的文本
    """
    return text[::-1]

def is_palindrome(text: str) -> bool:
    """
    检查是否为回文

    Args:
        text: 文本

    Returns:
        是否为回文
    """
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]

def extract_urls(text: str) -> List[str]:
    """
    提取文本中的URL

    Args:
        text: 文本

    Returns:
        URL列表
    """
    import re
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)

def mask_email(email: str, mask_char: str = "*") -> str:
    """
    遮蔽邮箱地址

    Args:
        email: 邮箱地址
        mask_char: 遮蔽字符

    Returns:
        遮蔽后的邮箱
    """
    if '@' not in email:
        return email

    local, domain = email.split('@', 1)

    if len(local) <= 2:
        masked_local = mask_char * len(local)
    else:
        masked_local = local[0] + mask_char * (len(local) - 2) + local[-1]

    return f"{masked_local}@{domain}"