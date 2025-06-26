import re


def space_pascal_case(text: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z][a-z])", " ", text)


def fetch_number(text: str) -> int:
    return int(re.search(r"[\d]+", text).group())
