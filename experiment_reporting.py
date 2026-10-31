"""Shared Stage1/Stage2 measurement schema; no changes to model or scoring.

FLOPs are a profiler-observed partial operator count, never a claimed complete
model cost. Historical measurements unavailable in a checkpoint stay null.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

SCHEMA_VERSION = "reaseg-report-v1"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parameter_report(model, optimizer=None):
    raw = model.module if hasattr(model, "module") else model
    rows = [dict(name=name, shape=list(p.shape), numel=p.numel(), requires_grad=p.requires_grad)
            for name, p in raw.named_parameters()]
    total = sum(row["numel"] for row in rows)
    trainable = sum(row["numel"] for row in rows if row["requires_grad"])
    groups = []
    if optimizer is not None:
        for index, group in enumerate(optimizer.param_groups):
            groups.append(dict(name=group.get("name", str(index)), lr=group["lr"],
                               params=sum(p.numel() for p in group["params"])))
    return dict(schema_version=SCHEMA_VERSION, total_params=total, trainable_params=trainable,
                frozen_params=total - trainable, trainable_ratio_percent=100 * trainable / total if total else 0,
                counting="unique Parameter objects; shared/tied tensors counted once",
                parameters=rows, optimizer_groups=groups)


def training_metadata(checkpoint_path, checkpoint=None):
    parent = Path(checkpoint_path).parent
    reporting = (checkpoint or {}).get("extra", {}).get("reporting", {})
    parameters = reporting.get("parameters")
    source = "checkpoint.extra.reporting" if parameters else None
    sidecar = parent / "trainable_parameters.json"
    if parameters is None and sidecar.is_file():
        parameters = json.loads(sidecar.read_text(encoding="utf-8"))
        source = str(sidecar)
    efficiency_path = parent / "efficiency.json"
    efficiency = json.loads(efficiency_path.read_text(encoding="utf-8")) if efficiency_path.is_file() else None
    return dict(parameters=parameters, efficiency=efficiency, source=source,
                efficiency_source=str(efficiency_path) if efficiency is not None else None,
                note="Training cost describes measured run/segments, not inference requires_grad; old unmeasured values remain null.")


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device):
    cuda = torch.device(device).type == "cuda"
    return dict(device=str(device),
                peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None,
                peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 2**30 if cuda else None)


class ComputeProfiler:
    def __init__(self, enabled, device, phase):
        self.enabled, self.device, self.phase = enabled, str(device), phase
        self.result = None

    @contextmanager
    def capture(self, batch_size, **metadata):
        if not self.enabled or self.result is not None:
            yield
            return
        # CPU operator records provide supported-op FLOPs without CUPTI/GPU
        # tracing. Operator coverage is incomplete, especially fused kernels.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    record_shapes=True, with_flops=True) as profiler:
            yield
        operators = [{"operator": event.key, "calls": event.count, "flops": int(event.flops)}
                     for event in profiler.key_averages() if event.flops]
        measured = sum(row["flops"] for row in operators)
        self.result = dict(status="partial_supported_ops", phase=self.phase, batch_size=int(batch_size),
                           device=self.device, supported_ops_flops_per_batch=measured,
                           supported_ops_flops_per_sample=measured / batch_size,
                           operators=operators, input=metadata)

    def report(self):
        result = dict(self.result or {"status": "disabled" if not self.enabled else "no_batch_measured",
                                     "phase": self.phase, "device": self.device})
        result.update(schema_version=SCHEMA_VERSION, full_model_flops=None, full_model_macs=None,
                      scope="One observed batch; profiler-supported operators only. Not a complete model FLOP/MAC estimate.",
                      limitations="Fused attention, custom scans, elementwise operations and some backward ops are not counted. Generation cost depends on actual sequence length. FLOPs are not divided by two to invent MACs.")
        return result


class TrainingReporter:
    """Call on every rank; persist canonical timing/parameter files on rank zero."""
    def __init__(self, out_dir, model, optimizer, args, rank, world_size, device, history):
        self.out_dir, self.rank, self.world_size = Path(out_dir), rank, world_size
        self.device, self.args, self.history = device, args, history
        self.parameters = parameter_report(model, optimizer)
        self.profiler = ComputeProfiler(args.profile_compute and rank == 0, device, "training_forward_loss_backward_optimizer")
        self.session_start = time.perf_counter()
        self.first_epoch = None
        if rank == 0:
            write_json(self.out_dir / "trainable_parameters.json", self.parameters)
            write_json(self.out_dir / "compute_profile.json", self.profiler.report())

    def start_epoch(self, epoch):
        synchronize(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.epoch = epoch
        if self.first_epoch is None:
            self.first_epoch = epoch
        self.epoch_start = self.phase_start = time.perf_counter()
        self.record = {"rank": self.rank}

    def finish_train(self, samples):
        synchronize(self.device)
        self.record.update(train_time_sec=time.perf_counter() - self.phase_start,
                           train_samples=int(samples), train_memory=memory_snapshot(self.device))
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.phase_start = time.perf_counter()

    def finish_validation(self, performed):
        synchronize(self.device)
        self.record.update(validation_time_sec=time.perf_counter() - self.phase_start if performed else 0.,
                           validation_memory=memory_snapshot(self.device) if performed else None)
        self.phase_start = time.perf_counter()

    def finish_epoch(self, log):
        synchronize(self.device)
        self.record.update(checkpoint_and_wait_time_sec=time.perf_counter() - self.phase_start,
                           epoch_wall_time_sec=time.perf_counter() - self.epoch_start)
        records = [None] * self.world_size
        if self.world_size > 1:
            dist.all_gather_object(records, self.record)
        else:
            records[0] = self.record
        training = max(row["train_time_sec"] for row in records)
        validation = max(row["validation_time_sec"] for row in records)
        samples = sum(row["train_samples"] for row in records)
        log.update(train_time_sec=training, validation_time_sec=validation,
                   checkpoint_time_sec=max(row["checkpoint_and_wait_time_sec"] for row in records),
                   epoch_wall_time_sec=max(row["epoch_wall_time_sec"] for row in records),
                   training_samples_processed=samples, training_samples_per_sec=samples / training if training else None,
                   efficiency_per_rank=records)
        if self.rank == 0:
            # history includes this log object. Old epochs without measurements
            # remain explicitly unmeasured rather than assigning zero cost.
            measured = [item for item in self.history if "efficiency_per_rank" in item]
            total_train = sum(item["train_time_sec"] for item in measured)
            total_val = sum(item["validation_time_sec"] for item in measured)
            total_save = sum(item["checkpoint_time_sec"] for item in measured)
            total_wall = sum(item["epoch_wall_time_sec"] for item in measured)
            memory = [m for item in measured for row in item["efficiency_per_rank"]
                      for m in (row["train_memory"], row["validation_memory"]) if m is not None]
            def peak(key):
                values = [m[key] for m in memory if m[key] is not None]
                return max(values) if values else None
            current_memory = [m for row in records for m in (row["train_memory"], row["validation_memory"])
                              if m is not None]
            for output, key in (("gpu_peak_allocated_gb", "peak_allocated_gb"),
                                ("gpu_peak_reserved_gb", "peak_reserved_gb")):
                values = [m[key] for m in current_memory if m[key] is not None]
                log[output] = max(values) if values else None
            completed = self.epoch + 1
            write_json(self.out_dir / "efficiency.json", dict(
                schema_version=SCHEMA_VERSION, phase="training", stage=getattr(self.args, "stage", getattr(self.args, "training_stage", "stage1")),
                world_size=self.world_size, batch_size_per_gpu=self.args.batch_size,
                model={k: self.parameters[k] for k in ("total_params", "trainable_params", "frozen_params")},
                completed_epochs=completed, history_records=len(self.history), measured_epochs=len(measured),
                unmeasured_epochs=completed - len(measured),
                full_run_measured=sorted(item["epoch"] for item in measured) == list(range(completed)),
                time=dict(training_sec=total_train, validation_sec=total_val, checkpoint_and_wait_sec=total_save,
                          measured_epoch_wall_sec=total_wall, current_session_wall_sec=time.perf_counter() - self.session_start,
                          gpu_hours=sum(item["epoch_wall_time_sec"] * item.get("world_size", self.world_size)
                                        for item in measured) / 3600,
                          scope="Epochs include data loading, training, validation and checkpoints; model/dataset initialization excluded. Distributed phase time is maximum over ranks; phase maxima need not sum to wall time."),
                memory=dict(max_peak_allocated_gb=peak("peak_allocated_gb"), max_peak_reserved_gb=peak("peak_reserved_gb")),
                epochs=measured))
            write_json(self.out_dir / "compute_profile.json", self.profiler.report())


def annotation_counts(mask_path):
    from experiment_protocol import read_mask
    mask=read_mask(mask_path)
    return dict(gt_foreground_pixels=int(mask.sum()),native_height=int(mask.shape[-2]),native_width=int(mask.shape[-1]))

def area_group(row):
    pixels=row['gt_foreground_pixels']
    if pixels==0: return 'empty_gt'
    ratio=pixels/(row['native_height']*row['native_width'])
    if ratio < .001: return '0-0.1%'
    if ratio < .005: return '0.1-0.5%'
    if ratio < .01: return '0.5-1%'
    return '>=1%'

def diagnostic_summary(rows):
    n=len(rows)
    result=dict(num_samples=n,near_zero_threshold=1e-6,low_dice_threshold=.1,
        near_zero_count=sum(r['dice']<=1e-6 for r in rows),low_dice_count=sum(r['dice']<.1 for r in rows),
        segmentation_failure_count=sum(bool(r.get('segmentation_failure',False)) for r in rows))
    for count,rate in [('near_zero_count','near_zero_rate'),('low_dice_count','low_dice_rate'),('segmentation_failure_count','failure_rate')]:
        result[rate]=result[count]/n if n else None
    for k in ('dice','iou','recall'):
        result['mean_'+k]=sum(r[k] for r in rows)/n if n else None
    return result

def diagnostic_groups(rows):
    groups=[]
    for dataset in ['ALL']+sorted({r['dataset'] for r in rows}):
        population=[r for r in rows if dataset=='ALL' or r['dataset']==dataset]
        for area in ('empty_gt','0-0.1%','0.1-0.5%','0.5-1%','>=1%'):
            selected=[r for r in population if area_group(r)==area]
            groups.append(dict(dataset=dataset,area_group=area,**diagnostic_summary(selected)))
    return groups

def save_diagnostic_outputs(directory, rows):
    import csv
    root=Path(directory)
    summary=diagnostic_summary(rows)
    dataset={d:diagnostic_summary([r for r in rows if r['dataset']==d]) for d in sorted({r['dataset'] for r in rows})}
    for filename,update in [('summary.json',summary),('per_dataset_summary.json',dataset)]:
        p=root/filename
        previous=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        if filename=='summary.json':
            # Preserve established floating point metric reductions exactly.
            previous.update({k:v for k,v in update.items() if k not in previous})
        else:
            for d,values in update.items():
                previous.setdefault(d,{})
                previous[d].update({k:v for k,v in values.items() if k not in previous[d]})
        write_json(p,previous)
    with (root/'area_group_summary.csv').open('w',encoding='utf-8',newline='') as handle:
        groups=diagnostic_groups(rows)
        writer=csv.DictWriter(handle,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    overall=root/'overall_summary.csv'
    if overall.exists():
        with overall.open(encoding='utf-8-sig',newline='') as handle:
            reader=csv.DictReader(handle); fields=list(reader.fieldnames); existing=list(reader)
        additions=('near_zero_count','near_zero_rate','low_dice_count','low_dice_rate')
        fields+= [k for k in additions if k not in fields]
        for row in existing:
            values=summary if row.get('dataset')=='ALL' else dataset[row['dataset']]
            row.update({k:values[k] for k in additions})
        with overall.open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(existing)

def gather_validation_rows(rows, distributed):
    if not distributed: return rows
    import torch.distributed as dist
    collected=[None]*dist.get_world_size()
    dist.all_gather_object(collected,rows)
    return [r for part in collected for r in part]
