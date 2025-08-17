



def investigate_text_alignment(content_text, segments, api_indices=None):
    """
    segment.textとcontent.textの一致箇所・前後の文字・改行や空白の有無を調査
    APIのstart_index/end_index（バイト単位）で正しく切り出せるかも検証
    api_indices: [(start_index, end_index), ...]  # バイト単位
    """
    def byte_index_to_str_index(text, byte_index):
        encoded = text.encode('utf-8')
        if byte_index >= len(encoded):
            return len(text)
        sub = encoded[:byte_index]
        return len(sub.decode('utf-8', errors='ignore'))

    for idx, seg in enumerate(segments):

        # API index検証
        if api_indices and idx < len(api_indices):
            api_start, api_end = api_indices[idx]
            str_start = byte_index_to_str_index(content_text, api_start)
            str_end = byte_index_to_str_index(content_text, api_end)
            api_slice = content_text[str_start:str_end]
            print(f"API index: start={api_start} end={api_end} (str_start={str_start} str_end={str_end})")
            print(f"APIで切り出した: '{api_slice}'")
            print(f"API index一致: {api_slice == seg}")
        print("-"*40)