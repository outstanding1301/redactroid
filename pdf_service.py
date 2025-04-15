import re

import pymupdf

from models import Pii


def extract_text(pdf) -> str:
    doc = pymupdf.open(stream=pdf)
    text = ''
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def redact(pdf, pii: Pii) -> bytes:
    doc = pymupdf.open(stream=pdf)
    texts = pii.get_texts()
    for page in doc:
        raw_dict = page.get_text("rawdict")
        for block in raw_dict["blocks"]:
            if block.get("type", 1) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if "chars" not in span:
                        continue
                    chars = span["chars"]
                    span_text = ''.join([char.get('c', '') for char in chars])
                    for target in texts:
                        for match in re.finditer(re.escape(target), span_text):
                            start, end = match.span()
                            union_rect = None
                            for char in chars[start:end]:
                                char_rect = pymupdf.Rect(char["bbox"])
                                if union_rect is None:
                                    union_rect = char_rect
                                else:
                                    union_rect |= char_rect
                            if union_rect:
                                print(f"Found '{target}' in span [{start}-{end}] - redaction annotation: {union_rect}")
                                page.add_redact_annot(union_rect, fill=(0, 0, 0))
        # 페이지에 적용
        page.apply_redactions()
    return doc.write()
