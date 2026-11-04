"""CPU tests for the portable TriCue graph; no dataset or trained weights required."""
import copy
import io
import unittest
import torch
from ultralytics.nn.tasks import DetectionModel
from rgbt.config import ARCHITECTURE
from rgbt.model import inject, MASKS, HFP
from rgbt.operators import FSPR, RCMF
from rgbt.protocol import Map50Stopper, DEFAULTS
from rgbt.runtime import ReproTrainer


class StopperTests(unittest.TestCase):
    def test_strict_map50_and_ties(self):
        stop=Map50Stopper(50)
        for epoch in range(1,51):
            stop.metric=.5
            self.assertFalse(stop(epoch,fitness=epoch))
        self.assertEqual(stop.best_epoch,1)
        self.assertTrue(stop(51,fitness=1000))
        stop.metric=.6
        self.assertFalse(stop(52,fitness=0))
        self.assertEqual(stop.best_epoch,52)

    def test_disabled_and_resume(self):
        a=Map50Stopper(0)
        for epoch in range(1,100):
            a.metric=.5
            self.assertFalse(a(epoch,fitness=0))
        b=Map50Stopper(0);b.__dict__.update(copy.deepcopy(a.__dict__))
        b.metric=.4
        self.assertFalse(b(100,fitness=100))
        self.assertEqual(b.best_epoch,1)
        self.assertEqual(DEFAULTS['rgbt'],dict(epochs=80,patience=0))
        self.assertEqual(DEFAULTS['vtuav'],dict(epochs=200,patience=50))
        with self.assertRaises(FloatingPointError):
            b.metric=float('nan');b(101)


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(0)
        cls.base=DetectionModel(copy.deepcopy(ARCHITECTURE),verbose=False)
        cls.x=torch.randn(1,6,128,128)

    def build(self,variant):
        return inject(copy.deepcopy(self.base),variant).eval()

    def test_all_masks_initial_add_and_shared_initialization(self):
        add=self.build('ADD');full=self.build('C')
        with torch.no_grad():expected=add(self.x)[0]
        for variant,mask in MASKS.items():
            with self.subTest(variant=variant):
                model=self.build(variant)
                with torch.no_grad():torch.testing.assert_close(model(self.x)[0],expected,rtol=0,atol=0)
                for group,slots in enumerate([(9,11,14,16,20,22),(23,24,25),(32,35,38)]):
                    for slot in slots:self.assertEqual(hasattr(model.model[slot],'gamma'),mask[group])
                common=full.state_dict()
                for name,value in model.state_dict().items():
                    if name in common:torch.testing.assert_close(value,common[name],rtol=0,atol=0)

    def test_uniform_modal_gate_and_frequency_reconstruction(self):
        model=self.build('C')
        r=torch.randn(2,64,8,8);t=torch.randn_like(r)
        mu=r;std=torch.ones_like(r)
        fusion=model.model[23].module
        self.assertIsInstance(fusion,RCMF)
        with torch.no_grad():
            w,condition=fusion.reliability(r,t,(mu,std),(t,std))
            torch.testing.assert_close(w,torch.full_like(w,.5),rtol=0,atol=0)
            self.assertEqual(condition.shape,(2,4,8,8))
            actual,_=fusion(r,t,(mu,std),(t,std))
            torch.testing.assert_close(actual,fusion.efc([r,t]),rtol=0,atol=0)
        hfp=HFP(64);x=torch.randn(2,64,8,8)
        h=hfp.high_pass(x);low=x-h
        torch.testing.assert_close(h+low,x,rtol=1e-5,atol=1e-6)
        branch=FSPR(hfp)
        torch.testing.assert_close(branch(x,None),hfp(x),rtol=0,atol=0)
        branch(x).square().mean().backward()
        self.assertTrue(torch.isfinite(branch.gate[-1].weight.grad).all())
        self.assertGreater(float(branch.gate[-1].weight.grad.abs().sum()),0)

    def test_save_reload_rebuild_and_no_batch_context(self):
        model=self.build('C')
        for slot in (9,11,14,16,20,22,23,24,25,32,35,38):
            model.model[slot].gamma.data.fill_(.1)
        buffer=io.BytesIO();torch.save(model,buffer);buffer.seek(0)
        loaded=torch.load(buffer,weights_only=False).eval()
        with torch.no_grad():
            first=model(self.x)[0]
            model(self.x*.3)
            torch.testing.assert_close(model(self.x)[0],first,rtol=0,atol=0)
            torch.testing.assert_close(loaded(self.x)[0],first,rtol=0,atol=0)
        class Owner:
            data={'nc':3}
        rebuilt=ReproTrainer.get_model(Owner(),cfg=model.yaml,weights=model,verbose=False).eval()
        with torch.no_grad():torch.testing.assert_close(rebuilt(self.x)[0],first,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
