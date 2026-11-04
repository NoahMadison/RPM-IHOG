"""Pin public harness task sources before evaluating new downstream results."""
import hashlib,json,pathlib,time,yaml,lm_eval
from huggingface_hub import HfApi
from lm_eval.api.task import ConfigurableTask
from common import PREFIX,ROOT,write_json,digest,journal
out=ROOT/'audit'/('downstream_sources_'+str(time.time_ns()));out.mkdir(exist_ok=False);configs=[]
package=pathlib.Path(lm_eval.__file__).parent
for name,relative in [('piqa','piqa/piqa.yaml'),('arc_easy','arc/arc_easy.yaml')]:
    path=package/'tasks'/relative;config=yaml.safe_load(path.read_text());info=HfApi().dataset_info(config['dataset_path']);config['dataset_kwargs']={'revision':info.sha}
    record={'task':name,'dataset':config['dataset_path'],'revision':info.sha,'harness_yaml_sha256':digest(path),'config':config}
    try:
        task=ConfigurableTask(config=config);split=task.eval_docs;h=hashlib.sha256();count=0
        for row in split:h.update(json.dumps(row,sort_keys=True,ensure_ascii=False).encode());h.update(b'\n');count+=1
        record.update(status='available',evaluation_documents=count,content_sha256=h.hexdigest(),split=task.config.test_split or task.config.validation_split)
    except Exception as e:record.update(status='blocked',error=type(e).__name__+': '+str(e))
    configs.append(record)
write_json(PREFIX/'configs'/('downstream_sources_'+out.name+'.json'),configs);journal('downstream_sources_pinned',sources=[{'task':c['task'],'revision':c['revision'],'status':c['status']} for c in configs]);print(json.dumps(configs,indent=2))
