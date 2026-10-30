"""Data-content guard outside the frozen training math, applied by the orchestrator."""
import json,pathlib
from common import PREFIX,digest

def canonical(data):
    return {key:data[key] for key in ['dataset','configuration','revision','tokenizer','tokenizer_revision','splits']}

def validate_identity(expected,actual):
    if expected!=actual:raise ValueError('training data/tokenizer revision or content changed; resume rejected')

def training_data_guard():
    from data_audit import latest_data
    data=latest_data();pin=json.loads((PREFIX/'configs/013_training_data_pin.json').read_text());identity=canonical(data);validate_identity(pin['identity'],identity)
    for split in ['train','validation','test']:
        if digest(pathlib.Path(data['artifact'])/(split+'.pt'))!=data['splits'][split]['tokens_sha256']:raise ValueError('corrupt token buffer; resume rejected')
    return {'identity':identity,'pin_sha256':digest(PREFIX/'configs/013_training_data_pin.json'),'actual_token_buffers_validated':True}
