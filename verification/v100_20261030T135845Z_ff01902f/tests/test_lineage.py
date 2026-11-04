import copy,unittest
from lineage import validate_identity,training_data_guard
class LineageTests(unittest.TestCase):
    def test_actual_pinned_data(self):self.assertTrue(training_data_guard()['actual_token_buffers_validated'])
    def test_changed_revision_and_content_rejected(self):
        original={'revision':'r1','tokenizer_revision':'t1','content_sha256':'c1'}
        for key in original:
            changed=copy.deepcopy(original);changed[key]+='different'
            with self.assertRaisesRegex(ValueError,'resume rejected'):validate_identity(original,changed)
if __name__=='__main__':unittest.main()
