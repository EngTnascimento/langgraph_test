import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
