"""Dependency-free multiplicity and scientific interpretation helpers."""
from __future__ import annotations
import math


def adjusted_pvalues(values, *, method='by'):
    """Benjamini–Yekutieli FDR adjustment; optional explicit BH."""
    if method not in {'bh', 'by'}:
        raise ValueError('method must be bh or by')
    ordered = sorted([(i, float(p)) for i,p in enumerate(values) if p is not None and math.isfinite(float(p)) and 0 <= float(p) <= 1], key=lambda x:x[1])
    n = len(ordered)
    harmonic = sum(1/i for i in range(1,n+1)) if method == 'by' else 1.
    output = [None]*len(values)
    minimum = 1.
    for rank in range(n,0,-1):
        index,p = ordered[rank-1]
        minimum = min(minimum, p*n*harmonic/rank)
        output[index] = minimum
    return output


def annotate_test_family(panel):
    """Add q-values across returned bivariate hypotheses, preserving raw p."""
    tests=[]
    def visit(value):
        if isinstance(value,dict):
            for key in ('p','p_value'):
                if key in value and isinstance(value[key],(int,float)):
                    tests.append((value,key))
            for child in value.values():
                if isinstance(child,(dict,list)):
                    visit(child)
        elif isinstance(value,list):
            for child in value:
                visit(child)
    visit(panel)
    qs=adjusted_pvalues([row[key] for row,key in tests])
    for (row,_),q in zip(tests,qs):
        row.update(q_value=q, significant_adjusted=q is not None and q<.05,
                   multiple_testing='benjamini_yekutieli', family_size=len(tests))
    return panel


def score_interpretation(analysis):
    critical=[f.get('code',f.get('title')) for f in analysis.get('findings',[]) if f.get('severity')=='critical']
    return {'method':'legacy_domain_penalties', 'calibrated':False, 'cross_modality_comparable':False,
            'probability':False, 'critical_findings':critical, 'critical_override':bool(critical),
            'interpretation':'Inspect critical findings regardless of aggregate score. Scores and beacon confidence are heuristic, not estimated probabilities; sparsity, rank and connectivity require domain context.'}
