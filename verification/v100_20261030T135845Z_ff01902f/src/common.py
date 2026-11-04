"""Task-owned writes, source identity, strict GPU selection, and evidence primitives."""
import datetime, hashlib, json, os, pathlib, subprocess, sys, time, uuid

PREFIX = pathlib.Path(os.environ.get('VERIFY_PREFIX', pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(os.environ.get('TASK_ROOT', PREFIX.parents[2]))

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body=json.dumps(obj,indent=2,allow_nan=False)
    with path.open('x') as f: f.write(body)

def journal(event, **data):
    record = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), event=event, **data)
    with (PREFIX/'events.jsonl').open('a') as f: f.write(json.dumps(record, allow_nan=False)+'\n')

def new_run(stage):
    name = f'{time.time_ns()}_{stage}_{uuid.uuid4().hex[:6]}'
    path = PREFIX/'results'/name
    path.mkdir(exist_ok=False)
    return path

def source_identity():
    return {'base': json.loads((PREFIX/'source_inventory.json').read_text())['base_sha'],
            'plan_sha256': digest(PREFIX/'EXPERIMENT_PLAN.json'),
            'sources': {str(p.relative_to(PREFIX)): digest(p) for p in sorted((PREFIX/'src').rglob('*')) if p.is_file()}}

def gpu_guard():
    import torch
    requested = json.loads((PREFIX/'source_inventory.json').read_text())['gpu_uuid']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('1', requested), 'visibility must select requested V100'
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,pci.bus_id,memory.total,compute_cap', '--format=csv,noheader,nounits'], text=True)
    rows = [r.split(', ') for r in inventory.strip().splitlines()]
    row = next(r for r in rows if r[2] == requested)
    assert 'V100' in row[1] and row[3].lower() == '00000000:06:00.0' and int(row[4]) == 32768 and row[5] == '7.0', row
    assert torch.cuda.device_count() == 1, 'only the selected GPU may be visible'
    prop = torch.cuda.get_device_properties(0)
    assert str(prop.uuid).replace('GPU-', '') == requested.replace('GPU-', '')
    assert (prop.major, prop.minor) == (7, 0) and 'V100' in prop.name
    assert 31*(1<<30) < prop.total_memory <= 32*(1<<30)
    assert (prop.pci_domain_id, prop.pci_bus_id, prop.pci_device_id) == (0,6,0)
    assert 'sm_70' in torch.cuda.get_arch_list(), 'PyTorch must contain Volta kernels'
    torch.cuda.set_device(0)
    return {'inventory':inventory,'torch_device':str(prop),'torch':torch.__version__, 'cuda':torch.version.cuda,
            'compiled_architectures':torch.cuda.get_arch_list(), 'python':sys.version}

def assert_resume(expected, actual):
    if expected != actual: raise ValueError('resume source/configuration identity mismatch')

def paired_ci(values):
    import numpy as np
    from scipy.stats import t
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    if len(a)<2: return {'mean':mean,'n':len(a),'ci95':None}
    d = float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/len(a)**.5)
    return {'mean':mean,'n':len(a),'ci95':[mean-d,mean+d]}

def block_bootstrap(deltas, seed=919, n=2000):
    import numpy as np
    x=np.asarray(deltas,float);rng=np.random.default_rng(seed)
    boot=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(n)])
    return {'mean':float(x.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'unit':'contiguous context block','n_blocks':len(x)}

def synchronize():
    import torch
    torch.cuda.synchronize()
