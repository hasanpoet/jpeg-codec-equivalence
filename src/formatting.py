"""Display-only formatting shared by manuscript values and table bodies."""


from decimal import Decimal


def format_number(value, specification):
    text = format(value, specification)
    if text.startswith('-') and Decimal(text.replace(',', '')) == 0:
        text = text[1:]
    return text
