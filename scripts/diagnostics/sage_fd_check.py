"""Diagnostic of published SAGE v2 pseudocode, not an author-code reproduction.

SAGE Algorithm 1: https://arxiv.org/html/2510.02470v2
FD reference: https://arxiv.org/html/1501.01711v2
Only NumPy is required. No data downloads, training or network calls.
"""
from pathlib import Path
import json
import numpy as np

def shrink(s):
    _, singular, vt = np.linalg.svd(s, full_matrices=False)
    return np.sqrt(np.maximum(singular**2-singular[-1]**2,0))[:,None]*vt

def sage_algorithm_1_literal(rows, ell):
    s=np.zeros((ell,rows.shape[1]),dtype=np.float64)
    for r,g in enumerate(rows):
        s[r % ell]=g
        if (r+1) % ell==0:
            s=shrink(s)
    return s

def basic_frequent_directions(rows, ell):
    # Simple FD, zero last row after every shrink. This is a correctness
    # reference, not the fast buffered version or a performance benchmark.
    s=np.zeros((ell,rows.shape[1]),dtype=np.float64)
    for g in rows:
        s[-1]=g
        s=shrink(s)
    return s

def run():
    rows=np.array([[10.,0.,0.],[10.,0.,0.],[0.,1.,0.],[0.,1.,0.]])
    ell,rank=2,1
    gram=rows.T@rows
    singular=np.linalg.svd(rows,compute_uv=False)
    residual=float(np.sum(singular[rank:]**2))
    bound=residual/(ell-rank)
    literal=sage_algorithm_1_literal(rows,ell)
    proper=basic_frequent_directions(rows,ell)
    err_literal=float(np.linalg.norm(gram-literal.T@literal,ord=2))
    err_fd=float(np.linalg.norm(gram-proper.T@proper,ord=2))
    assert np.allclose(gram,np.diag([200.,2.,0.]))
    assert np.allclose(literal.T@literal,np.diag([0.,2.,0.]))
    assert np.allclose(proper.T@proper,np.diag([198.,0.,0.]))
    assert err_literal>bound+1e-8
    assert err_fd<=bound+1e-8
    # A separate illustration: cosine agreement of NORMALIZED gradients
    # does not ensure closeness to the mean of UNNORMALIZED gradients.
    g=np.array([[1.,0.],[1.,0.],[0.,100.]])
    unit=g/np.linalg.norm(g,axis=1,keepdims=True)
    u=unit.mean(axis=0);u/=np.linalg.norm(u)
    scores=unit@u
    chosen=np.argsort(-scores,kind='stable')[:2]
    full=g.mean(axis=0); selected=g[chosen].mean(axis=0)
    cosine=float(full@selected/(np.linalg.norm(full)*np.linalg.norm(selected)))
    return {'checked_on':'2026-10-12','scope':'Literal pseudocode diagnostic; no authors code or training reproduction',
      'numpy_version':np.__version__,'ell':ell,'rank':rank,'rows':rows.tolist(),
      'full_gram':gram.tolist(),'literal_sage_gram':(literal.T@literal).tolist(),
      'reference_fd_gram':(proper.T@proper).tolist(),'fd_error_bound':bound,
      'literal_spectral_error':err_literal,'reference_spectral_error':err_fd,
      'literal_violates_fd_bound':True,
      'normalized_consensus_illustration':{'gradients':g.tolist(),'selected_indices':chosen.tolist(),
       'selected_vs_full_raw_gradient_cosine':cosine,'interpretation':'No guarantee of raw-mean gradient matching follows from directional agreement alone.'}}

if __name__=='__main__':
    result=run()
    target=Path(__file__).with_name('sage-diagnostics.json')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
