"""Independent small-tree checks of exact families and minimax hierarchies."""
import argparse,json
from pathlib import Path
import numpy as np
from exact_family_audit import enumerate_paths,possible_masks
from validate_bounds import all_trees,possible_tree
from referee_revision_study import minimax,hierarchy_interval
from network_bounds import complete_edges,mst


def main(out):
    rng=np.random.default_rng(20261026);checks=0
    for p in [3,4,5]:
        edges=complete_edges(p);trees=all_trees(p,edges);te,paths,masks=enumerate_paths(p)
        assert set(map(int,masks))=={sum(1<<x for x in s) for s in trees}
        for rep in range(10):
            lo=rng.integers(0,4,len(edges)).astype(float);hi=lo+rng.integers(0,3,len(edges))
            expected={sum(1<<x for x in s) for s in trees if possible_tree(s,trees,lo,hi)}
            assert expected==set(map(int,possible_masks(te,paths,masks,lo,hi)))
            w=rng.uniform(0,2,len(edges));tree=mst(p,edges,w)
            assert np.allclose(minimax(p,[edges[x] for x in tree],w[list(tree)]),minimax(p,edges,w))
            checks+=1
    for _ in range(100):
        p=6;edges=complete_edges(p)
        lo=[rng.uniform(0,1,len(edges)) for s in [0,1]]
        hi=[a+rng.uniform(0,1,len(edges)) for a in lo]
        w=[rng.uniform(a,b) for a,b in zip(lo,hi)]
        lower,upper=hierarchy_interval(p,edges,lo,hi)
        d=float(np.abs(minimax(p,edges,w[0])-minimax(p,edges,w[1])).max())
        assert lower<=d+1e-12 and d<=upper+1e-12
        assert d<=np.max(np.abs(w[0]-w[1]))+1e-12
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(status='passed',seed=20261026,independent_exact_family_and_tree_hierarchy_checks=checks,
                                  hierarchy_box_and_lipschitz_checks=100),indent=2))
    print(out.read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
