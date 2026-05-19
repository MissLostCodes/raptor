import fitz
import tiktoken
def extract_txt_pages(txt_path):
    print("Extracting text pages...")
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [
        {
            "page_num": 1,
            "text": text
        }
    ]

def extract_document_pages(file_path):
    print("Extracting file pages...extract_document_pages")
    if file_path.endswith(".txt"):
        return extract_txt_pages(file_path)

    elif file_path.endswith(".pdf"):
        return extract_pdf_pages(file_path)

    else:
        raise ValueError("Unsupported file type")

def extract_pdf_pages(pdf_path):
    """
    Extract pages from a pdf file.
    :param pdf_path:
    :return:
    """
    print("Extracting pages...extract_pdf_pages(pdf_path) called")
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_num": i + 1,
            "text": text
        })
    print(f"{len(pages)} pages extracted.")
    return pages


def count_tokens(text):
    print("Counting tokens...count_tokens(text) called ")
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def build_page_text(pages):
    """
    Build text from a list of pages.
    :param pages:
    :return:
    """
    print("Building text...build_page_text(pages) called ")
    result = ""
    for page in pages:
        result += f"\n<page_{page['page_num']}>\n"
        result += page["text"]
        result += "\n"
    print(f"{len(pages)} pages built.")
    return result