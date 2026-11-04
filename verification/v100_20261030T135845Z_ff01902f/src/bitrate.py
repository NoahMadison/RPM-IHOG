import json,math,pathlib,torch
from common import ROOT,PREFIX,write_json,digest
from math_codec import encode,decode,tensor_bytes,materialize,spectra

def main(out,args):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    ids=json.loads(sorted((PREFIX/'results').glob('*_data_*/model_identities.json'))[-1].read_text());identity=ids[0]
    index=json.loads(pathlib.Path(hf_hub_download(identity['id'],'model.safetensors.index.json',revision=identity['revision'],local_files_only=True)).read_text())
    key=next(k for k in index['weight_map'] if 'layers.0.' in k and 'input_linear.weight' in k);path=hf_hub_download(identity['id'],index['weight_map'][key],revision=identity['revision'],local_files_only=True)
    with safe_open(path,framework='pt',device='cpu') as f:w=f.get_slice(key)[0,:512,:].float().to('cuda:0')
    artifact=ROOT/'artifacts'/('bitrate_'+out.name);artifact.mkdir(exist_ok=False);rows=[]
    for kind in ['pure','ternary','int2','int2_sign','int4']:
        p=encode(w,kind);rec=decode(p);bytes_=tensor_bytes(p);file=artifact/(kind+'.pt')
        with file.open('xb') as f:torch.save(p,f)
        # Ordinary low rank matched to actual tensor payload bytes (FP16 U/SV factors).
        rank=max(1,min(min(w.shape),bytes_//(2*sum(w.shape))))
        u,s,v=torch.linalg.svd(w,full_matrices=False);a=(u[:,:rank]*s[:rank]).half();b=v[:rank].half();lr=a.float()@b.float()
        sqnr=lambda r:10*math.log10(float(w.square().mean())/max(float((w-r).square().mean()),1e-30))
        rows.append({'kind':kind,'shape':list(w.shape),'actual_rank':p['meta'].get('rank',0),'tensor_payload_bytes':bytes_,'serialized_bytes':file.stat().st_size,'serialized_bpw':file.stat().st_size*8/w.numel(),'tensor_payload_bpw':bytes_*8/w.numel(),'weight_sqnr_db':sqnr(rec),'matched_payload_lowrank_rank':rank,'matched_payload_lowrank_bytes':tensor_bytes([a,b]),'matched_payload_lowrank_sqnr_db':sqnr(lr),'source_model':identity['id'],'source_revision':identity['revision'],'source_tensor':key,'scope':'complete gate matrix of expert 0, layer 0; not full-model fidelity'})
    # Direct native representation: exactly 4,352 scalars at rank 8 / 256x256.
    a=torch.randn(8,16,16)*(1/(8*256))**.25;b=torch.randn_like(a)*(1/(8*256))**.25;row=torch.ones(256)
    native={'a':a.half(),'b':b.half(),'row':row.half()};file=artifact/'native_direct.pt'
    with file.open('xb') as f:torch.save(native,f)
    original=materialize(a,b)*row[:,None];rec=materialize(native['a'].float(),native['b'].float())*native['row'].float()[:,None]
    write_json(out/'bitrate.json',{'actual_real_matrix_codecs':rows,'native_direct_synthetic_fixture':{'shape':[256,256],'rank':8,'learned_scalars':4352,'FP16_payload_bytes':tensor_bytes(native),'FP16_payload_bpw':tensor_bytes(native)*8/65536,'serialized_bytes':file.stat().st_size,'serialized_bpw':file.stat().st_size*8/65536,'rounding_relative_l2':float((rec-original).norm()/original.norm())},'legacy_shape_analytical':{'shape':[256,256],'rank':8,'generic_core_bpw':16*(8*(256+256)+512)/65536,'ideal_ternary_core_bpw':16*(8*512+512+256)/65536+math.log2(3),'actual_2bit_ternary_core_bpw_before_framing':16*(8*512+512+256)/65536+2,'int2_plus_sign_residual_bits':3},'dense_baselines':{'FP16_bytes':w.numel()*2,'FP32_bytes':w.numel()*4},'counting_notes':'unique backing storage including scales/levels; file framing measured separately; metadata and manifests included in full pretrained archive totals; direct native synthetic fixture independent from trained exports'})
