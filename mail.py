class Mail:
    def __init__(self, text):
        first_nl = text.find('\n\n')

        self.head = text[:first_nl]
        self.content = text[first_nl:]

if __name__ == '__main__':
    f = open('data/2/00163.2ceade6f8b0c1c342f5f95d57ac551f5', "r", encoding="utf-8")
    mail = Mail(f.read())
    print(mail.head)
    print('TADY KONCI HEAD')
    print(mail.content)