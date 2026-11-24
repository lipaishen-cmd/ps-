"""Register explicit PCB method IDs, then use the existing result-only evaluator."""
from . import METHODS, ROOT
import sys
sys.path.insert(0,str(ROOT/'baselines'))
from baselines import unified_evaluator as shared
shared.METHOD_FAMILIES=dict(shared.METHOD_FAMILIES,external_rl=tuple(METHODS.values()))
shared.KNOWN_METHODS=frozenset(m for group in shared.METHOD_FAMILIES.values() for m in group)
LEGACY_METHOD_IDS={'IDDQN-II-PCB':'ATTENTION-DDQN-PCB'}
original_canonical_method=shared.canonical_method
def canonical_method_with_legacy(value):
    return original_canonical_method(LEGACY_METHOD_IDS.get(str(value).strip().upper(),value))
shared.canonical_method=canonical_method_with_legacy
original_load_records=shared.load_records
def checked_load_records(paths):
    import csv,json
    for path in paths:
        path=__import__('pathlib').Path(path)
        if path.suffix.lower()=='.csv':
            with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        else: rows=list(shared._json_rows(json.loads(path.read_text(encoding='utf-8')),path))
        for row in rows:
            if row.get('method_id','').upper() in (*METHODS.values(),*LEGACY_METHOD_IDS):
                if row.get('evidence_status')!='ADAPTED_METHOD_NOT_PAPER_REPLICATION':
                    raise ValueError('External RL result missing full-evaluation evidence status; smoke is forbidden')
    return original_load_records(paths)
shared.load_records=checked_load_records
if __name__=='__main__': shared.main()
