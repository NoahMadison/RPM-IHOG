"""Full frozen downstream comparisons after an adapter smoke test passes."""
import json,pathlib,subprocess,time
from common import PREFIX,ROOT,write_json,journal

def main():
    directory=ROOT/'artifacts'/('downstream_campaign_'+str(time.time_ns()));directory.mkdir(exist_ok=False)
    specs=[(model,codec) for model,codecs in [('ibm-granite/granite-3.1-1b-a400m-base',['original','int4','int2_sign','distilled']),('HuggingFaceTB/SmolLM2-1.7B',['original','int4','int2_sign'])] for codec in codecs]
    write_json(directory/'specs.json',specs)
    for index,(model,codec) in enumerate(specs):
        cmd=['bash',str(PREFIX/'run.sh'),'--stage','downstream','--model',model,'--codec',codec]
        with (directory/f'job_{index:03d}.stdout').open('x') as out,(directory/f'job_{index:03d}.stderr').open('x') as err:r=subprocess.run(cmd,stdout=out,stderr=err)
        write_json(directory/f'job_{index:03d}.json',{'command':cmd,'returncode':r.returncode});journal('downstream_campaign_job_exit',model=model,codec=codec,returncode=r.returncode)
        print(model,codec,r.returncode,flush=True)
        if r.returncode:raise RuntimeError('downstream job failed; preserve completed evidence and inspect new error')
    journal('downstream_campaign_complete')

if __name__=='__main__':main()
