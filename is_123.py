"""Модуль для проверки, является ли введённое значение числом 123."""

def is_123(value: str) -> bool:
    """Возвращает True, если значение равно 123."""
    try:
        return int(value) == 123
    except ValueError:
        return False


if __name__ == "__main__":
    user_input = input("Введите число: ")
    if is_123(user_input):
        print("Да, это число 123.")
    else:
        print("Нет, это не число 123.")
