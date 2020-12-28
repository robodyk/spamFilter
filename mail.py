import re

class Mail:
    def __init__(self, text, filename):
        first_nl = text.find('\n\n')

        self.head = text[:first_nl]
        self.content = text[first_nl:]
        self.content_no_html = self.remove_html_tags(self.content)
        self.subject = self.get_subject()
        self.filename = filename
        self.content_no_html_wordcount = self.get_total_word_count(self.content_no_html)
        self.subject_wordcount = self.get_total_word_count(self.subject)


    def get_total_word_count(self, str):
        str = str.replace('.', '').replace(',', '').replace('!', '').replace('?', '').replace('\n', ' ')
        words = str.split()
        return len(words)if len(words) >0 else 1

    def get_word_count(self, str, word):
        return str.count(word)

    def count_html_tags(self, str):
        matches = re.findall('<\/.*?>', str)

        return len(matches)

    def remove_html_tags(self, str):
        # this can be used to find all HTML (opening) tags - matches = re.findall('(<(?:.|\n)*?>)', str)
        reg = re.compile('<(.|\n)*?>')
        str = str.replace('=\n', '\n')
        return re.sub(reg, '', str)

    def check_spam_status(self):
        status_tag_index = self.head.find('X-Spam-Status: ')
        if status_tag_index < 0:
            return 1
        if self.head[status_tag_index + 15] == 'N':
            return 0
        return 99999999999

    def get_subject(self):
        subject_start = self.head.find("Subject: ") + 9
        subject_end = self.head.find('\n', subject_start)
        return self.head[subject_start:subject_end]

    def get_caps_and_chars_vector(self, str):
        if str == '':
            return [3*[0], 12*[0]]
        # vraci 2 podvektory v listu, neni to uplne smooth, ale usetri to 1 iteraci skrz body
        # char = neni v abecede
        char_count_dict = {';': 0, '&': 0, '*': 0, '$': 0, '+': 0, '=': 0, '!': 0, '>': 0, '%': 0}
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
                [100*v/total_count for v in char_count_dict.values()] +
                [char_count / total_count, (char_chains_sum / char_chain_count) if char_chain_count > 0 else 0, longest_char_chain]]

    def get_words_counts(self):
        upper_content = self.content_no_html.upper()
        count_fraction = self.content_no_html_wordcount/100
        return [[ #jsem awful, dxx pridavam 12
            self.get_word_count(upper_content, 'GRANT') / count_fraction,
            self.get_word_count(upper_content, 'DOLLARS') / count_fraction,
            self.get_word_count(upper_content, 'OPPORTUNITY') / count_fraction,
            self.get_word_count(upper_content, 'INSURANCE') / count_fraction,
            self.get_word_count(upper_content, 'MILLION') / count_fraction,
            self.get_word_count(upper_content, 'RESULTS') / count_fraction,
            self.get_word_count(upper_content, 'VIAGRA') / count_fraction,
            self.get_word_count(upper_content, 'ORDER') / count_fraction,
            self.get_word_count(upper_content, 'ONLY') / count_fraction,
            self.get_word_count(upper_content, 'RECEIVE') / count_fraction,
            self.get_word_count(upper_content, 'BUSINESS') / count_fraction,
            self.get_word_count(upper_content, 'CLICK') / count_fraction,
            self.get_word_count(upper_content, '100%') / count_fraction,
            self.get_word_count(upper_content, '#1') / count_fraction,
            self.get_word_count(upper_content, 'FREE') / count_fraction,
            self.get_word_count(upper_content, 'SATISFIED') / count_fraction,
            self.get_word_count(upper_content, 'OFF') / count_fraction,
            self.get_word_count(upper_content, 'GET') / count_fraction,
            self.get_word_count(upper_content, 'AD') / count_fraction,
            self.get_word_count(upper_content, 'ALL') / count_fraction,
            self.get_word_count(upper_content, 'NEW') / count_fraction,
            self.get_word_count(upper_content, 'BARGAIN') / count_fraction,
            self.get_word_count(upper_content, 'BONUS') / count_fraction,
            self.get_word_count(upper_content, 'BEST') / count_fraction,
            self.get_word_count(upper_content, 'PRICE') / count_fraction,
            self.get_word_count(upper_content, 'MOST') / count_fraction,
            self.get_word_count(upper_content, 'VIRUS') / count_fraction,
            self.get_word_count(upper_content, 'MONEY') / count_fraction,
            self.get_word_count(upper_content, '.JPG') / count_fraction,
            self.get_word_count(upper_content, '.PNG') / count_fraction,
            self.get_word_count(upper_content, '.COM') / count_fraction,
            self.get_word_count(upper_content, '.NET') / count_fraction,
        ]]

    def get_html_tags_count(self):
        upper_content = self.content.upper()
        count_fraction = self.content_no_html_wordcount / 100
        return [[
            self.get_word_count(upper_content, '<META') / count_fraction,
            self.get_word_count(upper_content, '<TITLE') / count_fraction,
            self.get_word_count(upper_content, '<HEAD') / count_fraction,
            self.get_word_count(upper_content, '<BODY') / count_fraction,
            self.get_word_count(upper_content, '<CONTENT') / count_fraction,
            self.get_word_count(upper_content, '<P') / count_fraction,
            self.get_word_count(upper_content, '<IMG') / count_fraction,
            self.get_word_count(upper_content, '<B>') / count_fraction,
            self.get_word_count(upper_content, '<BR>') / count_fraction,
            self.get_word_count(upper_content, '<TR') / count_fraction,
            self.get_word_count(upper_content, '<TD') / count_fraction,
            self.get_word_count(upper_content, '<UL') / count_fraction,
            self.get_word_count(upper_content, '<LI') / count_fraction,
        ]]

    def get_feature_vector_plr(self):
        vector = self.get_caps_and_chars_vector(self.subject) + \
                 self.get_caps_and_chars_vector(self.content_no_html) + \
                 self.get_words_counts() + \
                 self.get_html_tags_count()
        return vector

    def get_feature_vector_lr(self):
        temp = self.get_caps_and_chars_vector(self.subject)
        vector = temp[0] + temp[1]
        temp = self.get_caps_and_chars_vector(self.content_no_html)
        vector += temp[0] + temp[1] +\
            self.get_words_counts()[0] +\
            self.get_html_tags_count()[0]
        return vector

if __name__ == '__main__':
    f = open('data/1/0001.bfc8d64d12b325ff385cca8d07b84288', "r", encoding="utf-8")
    mail = Mail(f.read(), 'data/2/2490.f03277d54faea3974942b3213f38268f')
    print(mail.get_feature_vector_lr())
