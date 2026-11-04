import unittest,torch
from models import make_paired

class InitializationTests(unittest.TestCase):
    def test_dense_arm_initial_weights_identical(self):
        baseline=make_paired('adamw',81)
        for arm in ['row','tail','row_tail','legacy_blend']:
            other=make_paired(arm,81)
            self.assertEqual(set(baseline.state_dict()),set(other.state_dict()))
            for name,value in baseline.state_dict().items():self.assertTrue(torch.equal(value,other.state_dict()[name]),name)
    def test_native_lowrank_shared_and_matched_budget(self):
        base=make_paired('adamw',81);native=make_paired('native',81);lowrank=make_paired('lowrank',81)
        for name,value in base.state_dict().items():
            if '.moe.' not in name or '.router.' in name:
                self.assertTrue(torch.equal(value,native.state_dict()[name]),name)
                self.assertTrue(torch.equal(value,lowrank.state_dict()[name]),name)
        count=lambda m:sum(p.numel() for n,p in m.named_parameters() if '.moe.' in n and '.router.' not in n)
        self.assertEqual(count(native),count(lowrank));self.assertEqual(count(native),4*2*16*4352)
    def test_function_matched_native_experts(self):
        native=make_paired('native',81);dense=make_paired('adamw',81,function_matched=True)
        for a,b in zip(native.blocks,dense.blocks):
            for direction in ['up','down']:self.assertTrue(torch.equal(a.moe.weights(direction),b.moe.weights(direction)))

if __name__=='__main__':unittest.main()
