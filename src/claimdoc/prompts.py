"""Instructions per document type. The same text is used for zero-shot, fine-tuning and the OCR+LLM baseline,
so the comparison is about the model, not the prompt."""

RECEIPT = """Extract the receipt as JSON with exactly this structure (omit fields that are not printed):
{"menu": [{"nm": item name, "cnt": quantity, "unitprice": unit price, "price": line total, "sub": [{"nm": modifier, "price": price}]}],
 "sub_total": {"subtotal_price", "discount_price", "service_price", "tax_price", "etc"},
 "total": {"total_price", "cashprice", "changeprice", "creditcardprice", "emoneyprice", "menuqty_cnt", "menutype_cnt"}}
Copy every value exactly as printed (keep the original number formatting). Reply with JSON only."""

FORM = """Extract every filled-in field of this form as JSON:
{"headers": [section titles], "fields": [{"key": the printed label, "value": the filled-in text}]}
Use the label and value text exactly as written, in reading order. Skip empty fields. Reply with JSON only."""

PROMPTS = {"receipt": RECEIPT, "form": FORM}

OCR_LLM = """Below is OCR text of a scanned {doc} (reading order, may contain OCR errors).

{ocr}

{instruction}"""
