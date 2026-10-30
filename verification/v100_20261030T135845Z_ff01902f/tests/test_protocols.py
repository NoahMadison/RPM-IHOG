"""Meaningful failure-path tests runnable without any GPU or network."""
import io,json,pathlib,tempfile,unittest
from unittest.mock import patch
from streaming import RangeReader
from common import assert_resume,write_json

class Response(io.BytesIO):
    def __init__(self,status,content_range,body):super().__init__(body);self.status=status;self.headers={'Content-Range':content_range}

class ProtocolTests(unittest.TestCase):
    def test_reject_ignored_range(self):
        with patch('urllib.request.urlopen',return_value=Response(200,'',b'abcdefgh')):
            with self.assertRaisesRegex(ValueError,'ignored'):RangeReader('https://example.invalid').read(0,7)
    def test_reject_malformed_and_wrong_range(self):
        for cr in ['wrong','bytes 1-8/100','bytes 0-7/7']:
            with patch('urllib.request.urlopen',return_value=Response(206,cr,b'abcdefgh')):
                with self.assertRaises(ValueError):RangeReader('https://example.invalid').read(0,7)
    def test_reject_truncation_and_oversize(self):
        for body in [b'a',b'abcdefghijk']:
            with patch('urllib.request.urlopen',return_value=Response(206,'bytes 0-7/100',body)):
                with self.assertRaisesRegex(ValueError,'truncated|oversized'):RangeReader('https://example.invalid').read(0,7)
    def test_correct_counting(self):
        with patch('urllib.request.urlopen',return_value=Response(206,'bytes 0-7/100',b'abcdefgh')):
            r=RangeReader('https://example.invalid');self.assertEqual(r.read(0,7),b'abcdefgh');self.assertEqual(r.downloaded,8)
    def test_changed_resume_revision(self):
        with self.assertRaises(ValueError):assert_resume({'revision':'a'},{'revision':'b'})
    def test_reject_changed_source_size(self):
        a=Response(206,'bytes 0-7/100',b'abcdefgh');b=Response(206,'bytes 8-15/101',b'abcdefgh')
        with patch('urllib.request.urlopen',side_effect=[a,b]):
            reader=RangeReader('https://example.invalid');reader.read(0,7)
            with self.assertRaisesRegex(ValueError,'source size changed'):reader.read(8,15)
    def test_reject_changed_source_etag(self):
        a=Response(206,'bytes 0-7/100',b'abcdefgh');a.headers['ETag']='source1';b=Response(206,'bytes 8-15/100',b'abcdefgh');b.headers['ETag']='source2'
        with patch('urllib.request.urlopen',side_effect=[a,b]):
            reader=RangeReader('https://example.invalid');reader.read(0,7)
            with self.assertRaisesRegex(ValueError,'ETag changed'):reader.read(8,15)
    def test_transient_retry_counts(self):
        import urllib.error
        with patch('urllib.request.urlopen',side_effect=[urllib.error.URLError('transient'),Response(206,'bytes 0-7/100',b'abcdefgh')]):
            reader=RangeReader('https://example.invalid');reader.read(0,7);self.assertEqual(reader.requests[0]['attempt'],1);self.assertEqual(reader.downloaded,8)

if __name__=='__main__':unittest.main()
