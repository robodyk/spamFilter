import unittest
from mail import Mail

class TestMailHtml(unittest.TestCase):
    def test_count_html_tags(self):
        """Count HTML tags inside mail."""
        f = open('data/1/0419.a42a284750591b454968a76dfab38370', "r", encoding="utf-8")
        mail = Mail(f.read())
        f.close()

        self.assertEqual(77, mail.count_html_tags(mail.content),
                         'HTML tags count is incorrect.')

    def test_remove_html_tags(self):
        """Count HTML tags inside mail."""
        f = open('data/1/0419.a42a284750591b454968a76dfab38370', "r", encoding="utf-8")
        mail = Mail(f.read())
        f.close()

        self.assertEqual(-1, mail.remove_html_tags(mail.content).find('<IMG'),
                         'Not all HTML tags were removed')

        self.assertEqual(True, mail.remove_html_tags(mail.content).find('sitting in our database alone, which contains bank') != -1,
                         'Text was removed from mail')

if __name__ == "__main__":
    unittest.main()
