def normalize_phone(phone: str):
    return "".join(filter(str.isdigit, phone))
