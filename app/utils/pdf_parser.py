import fitz  # PyMuPDF


def extract_text_from_pdf(file):
    pdf = fitz.open(stream=file.read(), filetype="txt")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text