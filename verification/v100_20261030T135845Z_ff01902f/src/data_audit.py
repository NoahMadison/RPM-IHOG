import hashlib,json,pathlib,subprocess,urllib.request
import torch
from common import ROOT,PREFIX,write_json,digest

def main(out,args):
    from huggingface_hub import HfApi,hf_hub_download
    from datasets import load_dataset
    from transformers import AutoTokenizer
    api=HfApi();identities=[]
    for name in ['ibm-granite/granite-3.1-1b-a400m-base','HuggingFaceTB/SmolLM2-1.7B','Qwen/Qwen3.8-27B','Qwen/Qwen3.8-Flash-Next']:
        try:
            info=api.model_info(name);config=hf_hub_download(name,'config.json',revision=info.sha)
            identities.append({'id':name,'revision':info.sha,'config':json.loads(pathlib.Path(config).read_text()),'config_sha256':digest(config),'status':'available','files':[s.rfilename for s in info.siblings]})
        except Exception as e:
            identities.append({'id':name,'status':'source_unavailable','error':type(e).__name__+': '+str(e)[:800]})
    write_json(out/'model_identities.json',identities)
    info=api.dataset_info('Salesforce/wikitext');revision=info.sha
    tok_info=api.model_info('openai-community/gpt2');tok=AutoTokenizer.from_pretrained(tok_info.id,revision=tok_info.sha)
    ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=revision)
    artifact=ROOT/'artifacts'/('data_'+out.name);artifact.mkdir(exist_ok=False)
    meta={'dataset':'Salesforce/wikitext','configuration':'wikitext-2-raw-v1','revision':revision,'tokenizer':tok_info.id,'tokenizer_revision':tok_info.sha,'splits':{},'artifact':str(artifact)}
    for split in ['train','validation','test']:
        text='\n\n'.join(ds[split]['text']);ids=torch.tensor(tok.encode(text,add_special_tokens=False),dtype=torch.long)
        path=artifact/f'{split}.pt'
        with path.open('xb') as f:torch.save(ids,f)
        # Text bodies remain local; only content hashes/counts are published.
        meta['splits'][split]={'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'tokens':len(ids),'tokens_sha256':digest(path),'documents_or_lines':len(ds[split])}
    req=urllib.request.Request('https://api.github.com/repos/karpathy/char-rnn/commits/master',headers={'User-Agent':'RPM-IHOG-verification'})
    with urllib.request.urlopen(req,timeout=60) as response:commit=json.load(response)['sha']
    url=f'https://raw.githubusercontent.com/karpathy/char-rnn/{commit}/data/tinyshakespeare/input.txt'
    with urllib.request.urlopen(url,timeout=60) as response:body=response.read()
    if len(body)<1000001:raise ValueError('insufficient TinyShakespeare bytes')
    (artifact/'tinyshakespeare.txt').write_bytes(body)
    b=torch.tensor(list(body),dtype=torch.long);tokens=(b[:-1]*31+b[1:])%512
    with (artifact/'legacy_tokens.pt').open('xb') as f:torch.save(tokens[:1000000],f)
    meta['legacy']={'url':url,'revision':commit,'sha256':hashlib.sha256(body).hexdigest(),'train_tokens':900000,'validation_tokens':100000}
    write_json(out/'data.json',meta)
    print(json.dumps(meta,indent=2),flush=True)

def latest_data():
    matches=sorted((PREFIX/'results').glob('*_data_*/data.json'))
    if not matches: raise RuntimeError('run data stage first')
    return json.loads(matches[-1].read_text())
