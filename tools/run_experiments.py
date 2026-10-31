#!/usr/bin/env python3
"""Shared new-project launcher; --dry_run never imports or loads models."""
import argparse, json, os, shlex, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medspar.experiments import ExperimentConfig, EXPERIMENTS
NEG={'gradient_checkpointing':'--no-gradient_checkpointing','unfreeze_steer_modules':'--no-unfreeze_steer_modules',
     'train_lm_embeddings':'--no_train_lm_embeddings','train_lm_head':'--no_train_lm_head','profile_compute':'--no-profile_compute'}
def args_list(values):
    result=[]
    for k,v in values.items():
        if isinstance(v,bool):
            if v: result.append('--'+k)
            elif k in NEG: result.append(NEG[k])
        elif isinstance(v,list): result += ['--'+k,*map(str,v)]
        elif v is not None: result += ['--'+k,str(v)]
    return result
def output_name(stage,exp):
    return 'stage0_medsam' if stage=='stage0' else 'stage1_TC-MVE' if stage=='stage1' else 'stage2_MedSPAR' if exp=='main' else exp
def build(cli):
    filename={'stage0':'stage0_medsam.json','stage1':'stage1_tc_mve.json','stage2':'stage2_medspar.json'}[cli.stage]
    values=json.loads((ROOT/'configs'/filename).read_text())
    ids=cli.gpu_ids.split(',')
    if len(set(ids))!=len(ids) or any(not i.isdigit() for i in ids): raise ValueError('GPU_IDS must be distinct numeric IDs')
    values.update(seed=cli.seed,gpu_ids=cli.gpu_ids,num_gpus=len(ids))
    for k,v in [('data_root',cli.data_root),('epochs',cli.epochs),('batch_size',cli.batch_size),('val_batch_size',cli.val_batch_size)]:
        if v is not None: values[k]=v
    pre=Path(cli.pretrained_root)
    values['medsam_checkpoint']=str(pre/'medsam_vit_b.pth')
    if cli.stage=='stage2':
        policy=ExperimentConfig(cli.experiment)
        expected=json.loads((ROOT/'configs/experiments.json').read_text())[cli.experiment]
        if policy.flags()!=expected: raise ValueError('Experiment matrix and model flags disagree')
        values.update(experiment_id=cli.experiment,vision_type=policy.vision_backbone,
            llm_name_or_path=str(pre/'Qwen3.5-2B-SEG'),steer_text_encoder=str(pre/'bert-base-uncased'),
            stage1_checkpoint='' if cli.resume or cli.experiment=='medsam_vision' else str(Path(cli.work_root)/'stage1_TC-MVE/best.pth'),
            stage0_checkpoint='' if cli.resume or cli.experiment!='medsam_vision' else str(Path(cli.work_root)/'stage0_medsam/best.pth'))
        if not policy.spatial_supervision: values['lambda_attn']=0.
    else: values['text_encoder']=str(pre/'bert-base-uncased')
    directory=Path(cli.output_dir) if cli.output_dir else Path(cli.work_root)/output_name(cli.stage,cli.experiment)
    if cli.action=='train':
        values.update(output_dir=str(directory),resume=cli.resume or '',smoke_batches=cli.smoke_batches)
        entry={'stage0':'train/train_stage0_medsam.py','stage1':'train/train_stage1.py','stage2':'train/train_stage2.py'}[cli.stage]
        command=[sys.executable]
        if len(ids)>1: command += ['-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(ids)}']
        command += [str(ROOT/entry),*args_list(values)]
        selected=cli.gpu_ids
    else:
        ckpt=cli.checkpoint or str(Path(cli.work_root)/output_name(cli.stage,cli.experiment)/'best.pth')
        out=cli.output_dir or str(Path(cli.eval_root)/((('stage0_medsam' if cli.stage=='stage0' else 'stage1_TC-MVE') if cli.stage!='stage2' else cli.experiment)+ '_'+cli.split))
        values=dict(data_root=values['data_root'],datasets=values['datasets'],medsam_checkpoint=values['medsam_checkpoint'],
                    split=cli.split,batch_size=cli.eval_batch_size,num_workers=4,precision='bf16',output_dir=out,local_files_only=True)
        if cli.stage=='stage2':
            values.update(checkpoint=ckpt,experiment_id=cli.experiment,llm_name_or_path=str(pre/'Qwen3.5-2B-SEG'),
                steer_text_encoder=str(pre/'bert-base-uncased'),llm_torch_dtype='bf16',prompt_source='auto',max_samples=cli.max_samples)
            entry='evaluation/eval_stage2_medspar.py'
        else:
            values.update(steer_checkpoint=ckpt,text_encoder=str(pre/'bert-base-uncased'))
            entry='evaluation/eval_stage0_medsam.py' if cli.stage=='stage0' else 'evaluation/eval_stage1_tc_mve.py'
        selected=cli.eval_gpu_id or cli.eval_gpu_ids or os.getenv('EVAL_GPU_ID') or cli.gpu_ids
        eval_ids=selected.split(',')
        if len(set(eval_ids))!=len(eval_ids) or any(not i.isdigit() for i in eval_ids):
            raise ValueError('Evaluation GPU IDs must be distinct numeric IDs')
        command=[sys.executable]
        if len(eval_ids)>1:
            command += ['-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(eval_ids)}']
        command += [str(ROOT/entry),*args_list(values)]
        directory=Path(out)
    return command,values,selected,directory
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','eval'])
    p.add_argument('--stage',choices=['stage0','stage1','stage2'],required=True)
    p.add_argument('--experiment',choices=EXPERIMENTS,default='main')
    p.add_argument('--gpu_ids',default=os.getenv('GPU_IDS','0,1,2,3,4,5'))
    p.add_argument('--seed',type=int,default=int(os.getenv('SEED','2026')))
    p.add_argument('--data_root',default=os.getenv('DATA_ROOT','/data/ReaSeg/data'))
    p.add_argument('--pretrained_root',default=os.getenv('PRETRAINED_ROOT','/data/ReaSeg/pretrained'))
    p.add_argument('--work_root',default=os.getenv('WORK_ROOT',str(ROOT/'work_dir')))
    p.add_argument('--eval_root',default=os.getenv('EVAL_ROOT',str(ROOT/'output')))
    p.add_argument('--output_dir');p.add_argument('--checkpoint');p.add_argument('--resume')
    p.add_argument('--epochs',type=int,default=int(os.environ['EPOCHS']) if 'EPOCHS' in os.environ else None)
    p.add_argument('--batch_size',type=int);p.add_argument('--val_batch_size',type=int)
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--eval_gpu_ids',default=os.getenv('EVAL_GPU_IDS'),
                           help='Evaluation GPU list; default: GPU_IDS, same as training')
    selection.add_argument('--eval_gpu_id',default=None,
                           help='Explicit single-GPU evaluation (legacy CLI option)')
    p.add_argument('--eval_batch_size',type=int,default=2,help='Per-GPU evaluation batch size')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--smoke_batches',type=int,default=0);p.add_argument('--max_samples',type=int,default=0)
    p.add_argument('--dry_run',action='store_true')
    cli=p.parse_args()
    command,values,selected,directory=build(cli)
    print(f"GPU IDs: {selected}; processes: {len(selected.split(','))}",flush=True)
    print(json.dumps(values,indent=2));print('COMMAND: '+shlex.join(command),flush=True)
    if cli.dry_run:return
    if directory.exists() and any(directory.iterdir()) and not (cli.action=='train' and cli.resume):
        raise FileExistsError(f'Refusing to overwrite {directory}')
    if cli.resume and cli.action!='train':raise ValueError('Use checkpoint, not resume, for evaluation')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=selected,PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT)+os.pathsep+env.get('PYTHONPATH',''))
    subprocess.run(command,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
