def clean_text(text):
    import unicodedata
    import re
    text = unicodedata.normalize('NFC', text)
    # Remove Wikipedia artifacts [1], [2]...
    text = re.sub(r'\[\d+\]|\[source\]|\[cần dẫn nguồn\]', '', text)
    
    # RETAIN: Vietnamese letters, numbers, punctuation, and MATH CHARACTERS (+ = ^ * / % > <)
    text = re.sub(r'[^\w\s\d.,!?()\"\"áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵĐđ\-+=^*/%><]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text