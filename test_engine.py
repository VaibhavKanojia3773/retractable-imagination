import unittest
import numpy as np
if __package__:
    from .engine import Evidence, State, replay, plan
else:
    from engine import Evidence, State, replay, plan

class RetractionTests(unittest.TestCase):
    def test_multiple_and_idempotent(self):
        events=[Evidence('a',1,2),Evidence('b',1,-3),Evidence('c',2,4)]
        state=State(np.zeros(8))
        for e in events:state.add(e)
        state.retract(['a','a','c'])
        np.testing.assert_allclose(state.logits,replay(np.zeros(8),events,{'a','c'}))
        before=state.logits.copy();state.retract(['a']);np.testing.assert_array_equal(state.logits,before)

    def test_unknown_is_atomic(self):
        state=State(np.zeros(3));state.add(Evidence('a',0,1));before=state.logits.copy()
        with self.assertRaises(KeyError):state.retract(['a','missing'])
        np.testing.assert_array_equal(state.logits,before)

    def test_dependency_and_retention(self):
        state=State(np.full(8,-2.));state.add(Evidence('a',1,-1))
        a=state.forecast('affected',[0,1]);b=state.forecast('retained',[4,5])
        self.assertEqual(state.retract(['a']),{'affected'})
        self.assertIs(state.forecast('retained',[4,5]),b)
        self.assertNotEqual(state.forecast('affected',[0,1]),a)

    def test_random_batch_matches_recompute(self):
        rng=np.random.default_rng(17);events=[Evidence(str(i),int(rng.integers(0,32)),float(rng.normal())) for i in range(1000)]
        state=State(np.zeros(32))
        for e in events:state.add(e)
        ids=[str(i) for i in rng.choice(1000,130,replace=False)]
        state.retract(ids);np.testing.assert_allclose(state.logits,replay(np.zeros(32),events,ids),atol=1e-12)

    def test_duplicate_rejected(self):
        state=State(np.zeros(3));state.add(Evidence('a',0,1))
        with self.assertRaises(ValueError):state.add(Evidence('a',2,3))

    def test_static_cell_not_double_counted(self):
        state=State(np.full(4,-2.))
        self.assertEqual(state.forecast('a',[0,1])['survival'],state.forecast('b',[0,1,0])['survival'])

if __name__=='__main__':unittest.main()
