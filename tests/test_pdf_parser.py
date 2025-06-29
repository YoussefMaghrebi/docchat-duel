from langchain_app.utils import load_all_pdfs_from_folder

if __name__ == "__main__":
    folder = "docs"
    texts = load_all_pdfs_from_folder(folder)
    for filename, content in texts.items():
        print(f"\n===== {filename} =====")
        print(content[:500])  # show first 500 chars
