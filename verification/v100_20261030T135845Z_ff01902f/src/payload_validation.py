"""Reject unsupported numerical/metadata payloads before decode or GPU launch."""
import torch
def validate(payload):
    m=payload['meta']
    if m.get('version')!=1 or m.get('kind') not in ['pure','ternary','int2','int2_sign','int4']:raise ValueError('unsupported codec metadata')
    if len(m['shape'])!=2 or any(not isinstance(n,int) or n<=0 for n in m['shape']):raise ValueError('invalid matrix shape')
    for name,tensor in payload.items():
        if isinstance(tensor,torch.Tensor) and tensor.is_floating_point() and not bool(tensor.isfinite().all()):raise ValueError('nonfinite stored tensor '+name+'; unsupported storage dynamic range')
    o,i=m['shape']
    if m['kind']=='int4':
        if payload['codes'].dtype!=torch.uint8 or payload['codes'].numel()!=(payload['scales'].numel()*64+1)//2:raise ValueError('INT4 code/scale shape mismatch')
        if payload['scales'].numel()*64<o*i:raise ValueError('INT4 insufficient matrix coverage')
        return payload
    a,b=payload['a'],payload['b'];po,pi=m['padded_shape']
    if a.ndim!=3 or b.ndim!=3 or a.shape[0]!=b.shape[0] or a.shape[0]!=m['rank']:raise ValueError('factor rank mismatch')
    if a.shape[1]*b.shape[1]!=po or a.shape[2]*b.shape[2]!=pi or po<o or pi<i:raise ValueError('factor shape/coverage mismatch')
    if payload['row'].numel()!=po or payload['col'].numel()!=pi:raise ValueError('scale shape mismatch')
    if m['bits']:
        codes=payload['codes']
        if codes.dtype!=torch.uint8 or codes.numel()!=(po*pi*2+7)//8:raise ValueError('residual code shape mismatch')
        if payload['scales'].numel()!=po:raise ValueError('residual scale shape mismatch')
        if m['kind']=='ternary':
            if any(bool((((codes>>shift)&3)==3).any()) for shift in [0,2,4,6]):raise ValueError('reserved ternary code3')
        if m['kind']=='int2_sign' and (payload['sign_codes'].dtype!=torch.uint8 or payload['sign_codes'].numel()!=(po*pi+7)//8 or payload['gain'].numel()!=po):raise ValueError('sign/gain metadata mismatch')
    return payload
