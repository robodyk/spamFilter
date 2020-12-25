import re

class Mail:
    def __init__(self, text, filename):
        first_nl = text.find('\n\n')

        self.head = text[:first_nl]
        self.content = text[first_nl:]
        self.content_no_html = self.remove_html_tags(self.content)
        self.filename = filename

    def get_word_count(self, str, word):
        return str.count(word)

    def count_html_tags(self, str):
        matches = re.findall('<\/.*?>', str)

        return len(matches)

    def remove_html_tags(self, str):
        # this can be used to find all HTML (opening) tags - matches = re.findall('(<(?:.|\n)*?>)', str)
        reg = re.compile('<(.|\n)*?>')

        return re.sub(reg, '', str)

    def check_spam_status(self):
        status_tag_index = self.head.find('X-Spam-Status: ')
        if status_tag_index <0:
            return 0.5
        if self.head[status_tag_index + 15] == 'N':
            return 0
        return 1

    def get_subject(self):
        subject_start = self.head.find("Subject: ") + 9
        subject_end = self.head.find('\n', subject_start)
        return self.head[subject_start:subject_end]

    def get_caps_and_chars_vector(self, str):
        if str == '':
            return [3*[0], 11*[0]]
        # vraci 2 podvektory v listu, neni to uplne smooth, ale usetri to 1 iteraci skrz body
        # char = neni v abecede
        char_count_dict = {'#': 0, '@': 0, '*': 0, '$': 0, '-': 0, '=': 0, '!': 0, '>': 0}
        char_chain = False
        char_count = 0
        longest_char_chain = 0
        char_chain_count = 0
        char_chains_sum = 0
        chain = False
        longest_chain = 0
        caps_count = 0
        total_count = 0
        chain_count= 0
        caps_chains_sum = 0
        for char in str:
            total_count += 1
            if ord('Z') >= ord(char) >= ord('A'):
                caps_count += 1
                if chain:
                    chain_len += 1
                else:
                    chain_len = 1
                    chain = True
            elif ord('z') >= ord(char) >= ord('a'):
                if chain:
                    chain = False
                    caps_chains_sum += chain_len
                    chain_count += 1
                    if longest_chain < chain_len:
                        longest_chain = chain_len
            else:
                    try:
                        char_count_dict[char] +=1
                        char_count +=1
                        if char_chain:
                            char_chain_len +=1
                        else:
                            char_chain_len = 1
                            char_chain = True
                    except KeyError:
                        if char_chain:
                            char_chain = False
                            char_chain_count += 1
                            char_chains_sum += char_chain_len
                            if longest_char_chain < char_chain_len:
                                longest_char_chain = char_chain_len
        return [[caps_count / total_count, (caps_chains_sum / chain_count) if chain_count > 0 else 0, longest_chain],
                [v for v in char_count_dict.values()] +
                [char_count / total_count, (char_chains_sum / char_chain_count) if char_chain_count > 0 else 0, longest_char_chain]]

    def get_words_counts(self):
        upper_content = self.content_no_html.upper()

        return [[
            self.get_word_count(upper_content, '100%'),
            self.get_word_count(upper_content, '#1'),
            self.get_word_count(upper_content, 'FREE'),
            self.get_word_count(upper_content, 'SATISFIED'),
            self.get_word_count(upper_content, 'OFF'),
            self.get_word_count(upper_content, 'GET'),
            self.get_word_count(upper_content, 'AD'),
            self.get_word_count(upper_content, 'ALL'),
            self.get_word_count(upper_content, 'NEW'),
            self.get_word_count(upper_content, 'BARGAIN'),
            self.get_word_count(upper_content, 'BONUS'),
            self.get_word_count(upper_content, 'BEST'),
            self.get_word_count(upper_content, 'PRICE'),
            self.get_word_count(upper_content, 'MOST'),
            self.get_word_count(upper_content, 'VIRUS'),
            self.get_word_count(upper_content, 'MONEY'),
            self.get_word_count(upper_content, '.JPG'),
            self.get_word_count(upper_content, '.PNG'),
            self.get_word_count(upper_content, '.COM'),
            self.get_word_count(upper_content, '.NET'),
        ]]

    def get_html_tags_count(self):
        upper_content = self.content.upper()

        return [[
            self.get_word_count(upper_content, '<META'),
            self.get_word_count(upper_content, '<TITLE'),
            self.get_word_count(upper_content, '<HEAD'),
            self.get_word_count(upper_content, '<BODY'),
            self.get_word_count(upper_content, '<CONTENT'),
            self.get_word_count(upper_content, '<P'),
            self.get_word_count(upper_content, '<IMG'),
            self.get_word_count(upper_content, '<B>'),
            self.get_word_count(upper_content, '<BR>'),
            self.get_word_count(upper_content, '<TR'),
            self.get_word_count(upper_content, '<TD'),
            self.get_word_count(upper_content, '<UL'),
            self.get_word_count(upper_content, '<LI'),
        ]]

    def get_feature_vector_prototype(self):
        vector = self.get_caps_and_chars_vector(self.get_subject()) + \
                 self.get_caps_and_chars_vector(self.content_no_html) + \
                 [[self.check_spam_status()]] + \
                 self.get_words_counts() + \
                 self.get_html_tags_count()
        return vector

if __name__ == '__main__':
    f = open('data/2/2490.f03277d54faea3974942b3213f38268f', "r", encoding="utf-8")
    mail = Mail(f.read(), 'data/2/2490.f03277d54faea3974942b3213f38268f')
    print(mail.get_feature_vector_prototype())
