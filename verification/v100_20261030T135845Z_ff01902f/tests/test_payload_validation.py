import copy,unittest,torch
from math_codec import encode
from payload_validation import validate
class PayloadTests(unittest.TestCase):
    def test_reject_nonfinite_scale_and_factor(self):
        p=encode(torch.eye(16),'int2_sign')
        for name in ['a','row','col','scales','gain']:
            bad=copy.deepcopy(p);bad[name].flatten()[0]=float('inf')
            with self.assertRaisesRegex(ValueError,'nonfinite'):validate(bad)
    def test_reject_shape_and_reserved_ternary(self):
        p=encode(torch.eye(16),'ternary');validate(p)
        bad=copy.deepcopy(p);bad['meta']['rank']+=1
        with self.assertRaisesRegex(ValueError,'rank'):validate(bad)
        bad=copy.deepcopy(p);bad['codes'][0]=255
        with self.assertRaisesRegex(ValueError,'reserved'):validate(bad)
    def test_activation_scale_dynamic_range_rejection(self):
        p=encode(torch.eye(16),'int2_sign',balance=False,activation=torch.full((16,),1e-6))
        with self.assertRaisesRegex(ValueError,'dynamic range'):validate(p)
if __name__=='__main__':unittest.main()
