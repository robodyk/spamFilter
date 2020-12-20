import re

class Mail:
    def __init__(self, text):
        first_nl = text.find('\n\n')

        self.head = text[:first_nl]
        self.content = text[first_nl:]
        self.content_no_html = self.remove_html_tags(self.content)

    def get_word_count(self, str, word):
        return str.count(word)

    def count_html_tags(self, str):
        matches = re.findall('<\/.*?>', str)

        return len(matches)

    def remove_html_tags(self, str):
        # this can be used to find all HTML (opening) tags - matches = re.findall('(<(?:.|\n)*?>)', str)
        reg = re.compile('<(.|\n)*?>')

        return re.sub(reg, '', str)

if __name__ == '__main__':
    f = open('data/2/00163.2ceade6f8b0c1c342f5f95d57ac551f5', "r", encoding="utf-8")
    mail = Mail(f.read())
    print(mail.head)
    print('TADY KONCI HEAD')
    print(mail.content)