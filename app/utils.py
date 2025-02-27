def str_to_list(text: str):
    split_data = [line.strip() for line in text.splitlines() if line.strip()]
    return split_data