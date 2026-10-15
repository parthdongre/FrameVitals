"""Independent mathematical oracles; tolerances reflect public rounding/float64."""
import math
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from framevitals.analysis.tensor import tensor_metrics, analyze_tensor
from framevitals.deep_statistics_v2 import (_normality, _bootstrap_ci, _fit_best_distribution,
    _categorical_column_stats, _group_difference_test, _cramers_v)
from framevitals.streaming_sketches import NumpyLogQuantileSketch


@pytest.mark.parametrize('scale', [1.,1e200,1e-200])
def test_singular_scale(scale):
    m=tensor_metrics(np.diag([3.,0.])*scale)['matrix']
    assert m['rank']==1 and m['condition_number'] is None
    assert m['condition_status']=='singular_under_tolerance'
    assert m['stable_rank']==pytest.approx(1.)
    assert m['effective_rank']==pytest.approx(1.)


@pytest.mark.parametrize('diagonal', [[1.,1.,1.],[4.,2.,1.],[0.,0.,0.]])
def test_diagonal_oracle(diagonal):
    s=np.array(diagonal); m=tensor_metrics(np.diag(s))['matrix']
    assert m['rank']==np.count_nonzero(s)
    if s.max():
        assert m['stable_rank']==pytest.approx(sum(s*s)/s.max()**2,abs=1e-6)
        p=s*s/sum(s*s)
        assert m['effective_rank']==pytest.approx(math.exp(-sum(p*np.log(p))),abs=1e-6)
        assert m['condition_number']==pytest.approx(s.max()/s.min())


def test_submatrix_has_no_full_rank_claim():
    r=analyze_tensor(np.diag(np.tile([1.,0.,1.,0.],600)),depth='quick')
    m=r['tensor']['matrix']
    assert m['population']=='sampled_submatrix'
    assert m['stable_rank'] is None and m['stable_rank_ratio'] is None
    assert m['submatrix_stable_rank']>0
    assert m['sampling_method']=='uniform_without_replacement'
    assert not any(f['code']=='tensor.low_effective_dimension' for f in r['findings'])


def test_tail_nonfinite_is_detected():
    x=np.zeros(500001); x[499999]=np.nan
    r=analyze_tensor(x,depth='quick')
    assert r['tensor']['nonfinite']['nan_count']==1
    assert r['tensor']['nonfinite']['coverage']=='full'
    assert any(f['code']=='tensor.non_finite' for f in r['findings'])


def test_complex_not_silently_real():
    r=tensor_metrics(np.array([1+2j]))
    assert r['numeric'] is False and r['reason']=='complex_distribution_unsupported'


def test_constant_normality():
    assert _normality(pd.Series(np.ones(30)))['is_probably_normal'] is None


def test_bootstrap_valid_bounds():
    r=_bootstrap_ci(pd.Series(np.ones(30)),np.mean)
    assert not r['available'] or r['low']==r['high']==1.


def test_fit_likelihood_original_units():
    x=pd.Series(np.linspace(10.,20.,60)); r=_fit_best_distribution(x)
    for e in r['leaderboard']:
        expected=2*len(e['params'])-2*sum(getattr(stats,e['name']).logpdf(x,*e['params']))
        assert e['aic']==pytest.approx(expected,rel=1e-6,abs=1e-4)


def test_entropy_support():
    assert _categorical_column_stats(pd.Series(['a','b',None]*10))['entropy_ratio']==pytest.approx(1.,abs=1e-4)


def test_group_labels():
    r=_group_difference_test(pd.Series([0.]+[1.]*10+[9.]*10),pd.Series(['tiny']+['left']*10+['right']*10))
    assert r['groups']==['left','right']


def test_cramers_perfect_association():
    x=pd.Series(['a']*10+['b']*10)
    assert _cramers_v(x,x)['cramers_v']==pytest.approx(1.)


@pytest.mark.parametrize('alpha',[.01,.1,.5])
def test_quantile_bound(alpha):
    gamma=(1+alpha)/(1-alpha)
    for k in [-20,0,20]:
        for x in [gamma**k*(1+1e-12),gamma**(k+1)*(1-1e-12)]:
            for sign in [-1,1]:
                q=NumpyLogQuantileSketch(alpha,zero_threshold=0).update([sign*x]).quantile(.5)
                assert abs(q-sign*x)/x<=alpha+1e-12


def test_sketch_merge_rejects_mismatch():
    with pytest.raises(ValueError):
        NumpyLogQuantileSketch(.01).merge(NumpyLogQuantileSketch(.01000001))
    with pytest.raises(ValueError):
        NumpyLogQuantileSketch(zero_threshold=0).merge(NumpyLogQuantileSketch(zero_threshold=1e-10))


def test_graph_oracles():
    nx=pytest.importorskip('networkx')
    from framevitals.analysis.graph import _spectral_summary,_sample_shortest_paths,_budget
    s=_spectral_summary(nx.path_graph(4),nx,_budget('quick'))
    assert s['algebraic_connectivity']==pytest.approx(2-2*math.cos(math.pi/4),abs=1e-8)
    p=_sample_shortest_paths(nx.DiGraph([(0,1),(1,2)]),nx,source_count=3,weight=None)
    assert p['mean_distance']==pytest.approx(4/3,abs=1e-6)
    assert p['reachable_ratio']==.5 and p['coverage']=='full'
    assert p['population']=='reachable_ordered_nonself_pairs'


def test_relational_types_and_populations():
    from framevitals.analysis.relational import inspect_relational
    r=inspect_relational({'a':pd.DataFrame({'id':[1,2]}),'b':pd.DataFrame({'id':['1','2']})})
    assert r['relationships'][0]['estimated_join_rows_sample']==0
    r=inspect_relational({'a':pd.DataFrame({'id':[1,2]}),'b':pd.DataFrame({'id':[1]*9+[3]})})
    q=r['relationships'][0]
    assert q['child_reference_coverage']==.5 and q['right_row_coverage']==.9
    assert q['relationship_status']=='inferred_candidate'


def test_irregular_time_unavailable():
    from framevitals.time_series import detect_and_analyze_time_series
    df=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=41).delete(20),'value':np.sin(np.arange(40))})
    r=detect_and_analyze_time_series(df)
    assert not r['decomposition']['available'] and not r['autocorrelation']['available']


@pytest.mark.parametrize('method',['bh','by'])
def test_fdr_reference(method):
    from framevitals.scientific import adjusted_pvalues
    ps=np.array([.001,.02,.04,.2,.9])
    np.testing.assert_allclose(adjusted_pvalues(ps,method=method),stats.false_discovery_control(ps,method=method),rtol=1e-14)


@pytest.mark.parametrize('seed',range(6))
def test_sketch_merge_and_lower_quantile(seed):
    x=np.random.default_rng(seed).lognormal(0,4,2000)
    full=NumpyLogQuantileSketch().update(x)
    merged=NumpyLogQuantileSketch()
    for part in np.array_split(x,13):
        merged.merge(NumpyLogQuantileSketch().update(part))
    for q in [0,.01,.25,.5,.99,1]:
        exact=np.quantile(x,q,method='lower')
        assert merged.quantile(q)==full.quantile(q)
        assert abs(full.quantile(q)-exact)/exact<=.010000000001


@pytest.mark.parametrize('seed',range(5))
def test_moments_partition_against_scipy(seed,monkeypatch):
    from framevitals.analysis_state import NumericColumnState
    monkeypatch.setenv('FRAMEVITALS_BACKEND','numpy')
    x=np.random.default_rng(seed).standard_t(6,1024)
    merged=NumericColumnState()
    for part in np.array_split(x,17):
        merged=merged.merge(NumericColumnState.from_series(pd.Series(part)))
    assert merged.mean==pytest.approx(x.mean(),abs=1e-13)
    assert merged.variance==pytest.approx(x.var(ddof=1),rel=1e-12)
    assert merged.skewness==pytest.approx(stats.skew(x,bias=False),abs=1e-12)
    assert merged.kurtosis==pytest.approx(stats.kurtosis(x,bias=False),abs=1e-12)


def test_drift_adjustment_and_energy():
    from framevitals.drift_analysis import compare_datasets
    rng=np.random.default_rng(8); a=pd.DataFrame(rng.normal(size=(100,5)))
    b=a+1
    r=compare_datasets(a,b)
    rows=r['columns']
    for row in rows:
        assert row['ks_q_value']>=row['ks_p_value']
        assert row['energy_distance']==pytest.approx(stats.energy_distance(a[row['column']],b[row['column']]))


def test_sparse_contingency_no_inference():
    r=_cramers_v(pd.Series(['a','a','b','b']),pd.Series(['c','c','d','d']))
    assert r['p'] is None and not r['inference_available']
