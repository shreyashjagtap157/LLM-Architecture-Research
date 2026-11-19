#!/usr/bin/env python3
"""Offline validator for the LLM Architecture v9 research package.
A structural pass is a preflight only; it is not model or scientific validation.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, zipfile
from pathlib import Path
import yaml
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent

def load_yaml(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def fail(errors, message): errors.append(message)

def semantic_manifest_checks(data, errors):
    if not isinstance(data, dict):
        fail(errors, 'manifest root must be a mapping'); return
    def walk(value, path='manifest'):
        if isinstance(value, dict):
            for k, item in value.items(): walk(item, f'{path}.{k}')
        elif isinstance(value, list):
            for i, item in enumerate(value): walk(item, f'{path}[{i}]')
        elif isinstance(value, str) and ('REPLACE_ME' in value or value.strip() == 'UNSPECIFIED'):
            fail(errors, f'placeholder value remains at {path}')
    walk(data)
    for path in ['run_id','specification.sha256','source.commit','hardware.compute_device_model','software.environment_digest']:
        cur = data
        for part in path.split('.'):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is None or (isinstance(cur, str) and (not cur.strip() or 'REPLACE_ME' in cur)):
            fail(errors, f'missing/placeholder semantic field: {path}')
    model = data.get('model', {})
    if isinstance(model, dict):
        total, active = model.get('total_parameters'), model.get('active_parameters')
        if isinstance(total, (int,float)) and isinstance(active, (int,float)) and active > total:
            fail(errors, 'active_parameters cannot exceed total_parameters')
    src = data.get('source', {})
    if isinstance(src, dict) and src.get('dirty_tree') is True and not src.get('patch_sha256'):
        fail(errors, 'dirty source tree requires patch_sha256')
    hw=data.get('hardware', {})
    if isinstance(hw,dict):
        if hw.get('compute_device_type') == 'cpu_only' and hw.get('device_memory_type') != 'system_ram':
            fail(errors, 'cpu_only runs must identify device_memory_type=system_ram')
        if hw.get('compute_device_type') in {'accelerator_only','managed_accelerator'} and not (isinstance(hw.get('device_memory_per_device_gib'),(int,float)) and hw.get('device_memory_per_device_gib')>0):
            fail(errors, 'accelerator runs require positive device_memory_per_device_gib')
    sw=data.get('software', {})
    if isinstance(sw,dict) and sw.get('environment_kind') == 'container' and not sw.get('container_digest'):
        fail(errors, 'container environments require an immutable container_digest')
    rtype = data.get('run_type')
    datasets = data.get('dataset', []) if isinstance(data.get('dataset'), list) else []
    role_use={'training':'training','validation':'validation','test':'evaluation','benchmark':'evaluation','reward':'reward_model_training'}
    for d in datasets:
        if not isinstance(d,dict):
            fail(errors, 'dataset entries must be mappings'); continue
        required_use=role_use.get(d.get('role'))
        allowed=set(d.get('allowed_uses',[]))
        if required_use and required_use not in allowed:
            fail(errors, f"dataset '{d.get('name')}' role={d.get('role')} lacks allowed_uses={required_use}")
        if d.get('rights_review_status') != 'approved_for_declared_uses':
            fail(errors, f"dataset '{d.get('name')}' rights review is not approved for declared uses")
        if d.get('role') == 'validation' and 'model_selection' not in allowed:
            fail(errors, f"validation dataset '{d.get('name')}' lacks model_selection use approval")
    if rtype in {'pretraining','continued_pretraining','sft','rl'}:
        if not any(d.get('role') == 'training' for d in datasets if isinstance(d, dict)):
            fail(errors, 'training run requires at least one dataset with role=training')
        if rtype in {'continued_pretraining','sft','rl'}:
            training = data.get('training', {})
            if not training.get('parent_checkpoint_sha256') and not data.get('model', {}).get('parent_checkpoint_sha256'):
                fail(errors, f'{rtype} requires parent checkpoint SHA-256')
    if rtype == 'rl' and not any(d.get('role') == 'reward' for d in datasets if isinstance(d, dict)):
        fail(errors, 'rl run requires at least one dataset with role=reward')
    if rtype in {'inference','evaluation','ablation','speculative_decode'} and not any(d.get('role') in {'test','benchmark'} and 'evaluation' in set(d.get('allowed_uses',[])) for d in datasets if isinstance(d, dict)):
        fail(errors, f'{rtype} run requires a test/benchmark dataset approved for evaluation')
    evaluation = data.get('evaluation', {})
    metrics = evaluation.get('metrics', []) if isinstance(evaluation, dict) else []
    p = data.get('profile_id')
    ver = data.get('verification', {})
    if p in {'P8','P9'} and rtype != 'speculative_decode':
        fail(errors, f'{p} profile is reserved for run_type=speculative_decode')
    if p == 'P8':
        required = ['draft_model_revision','draft_geometry','draft_checkpoint_sha256','draft_tokenizer_sha256','draft_config_sha256','verifier_algorithm_revision','state_commit_policy_revision','proposal_probability_available','fidelity_claim']
        for k in required:
            if ver.get(k) in (None, ''): fail(errors, f'P8 verification.{k} is required')
        if ver.get('fidelity_claim') == 'exact_target_distribution' and ver.get('proposal_probability_available') is not True:
            fail(errors, 'P8 cannot claim exact target distribution without proposal_probability_available=true')
        if ver.get('fidelity_claim') not in {'exact_target_distribution','deterministic_target_rule','bounded_approximation','unverified'}:
            fail(errors, 'P8 fidelity_claim must be exact_target_distribution, deterministic_target_rule, bounded_approximation or unverified')
        if ver.get('fidelity_claim') in {'exact_target_distribution','deterministic_target_rule','bounded_approximation'} and ver.get('fidelity_test_result') != 'passed':
            fail(errors, 'P8 verified fidelity claims require a passed fidelity test; use unverified to record failed/not-run evidence')
        if ver.get('fidelity_claim') == 'bounded_approximation':
            if not ver.get('approximation_metric'):
                fail(errors, 'P8 bounded_approximation requires approximation_metric')
            if not isinstance(ver.get('approximation_bound'),(int,float)) or ver.get('approximation_bound') < 0:
                fail(errors, 'P8 bounded_approximation requires non-negative approximation_bound')
            else:
                observed=[m for m in metrics if isinstance(m,dict) and m.get('name')==ver.get('approximation_metric')]
                if not observed or not isinstance(observed[0].get('value'),(int,float)):
                    fail(errors, 'P8 bounded_approximation requires a numeric evaluation metric matching approximation_metric')
                elif observed[0]['value'] < 0:
                    fail(errors, 'P8 approximation metric must be non-negative')
                elif observed[0]['value'] > ver['approximation_bound']:
                    fail(errors, 'P8 measured approximation metric exceeds declared approximation_bound')
        if ver.get('target_checkpoint_sha256') and data.get('model',{}).get('checkpoint_sha256') != ver.get('target_checkpoint_sha256'):
            fail(errors, 'P8 verifier target checkpoint must match model.checkpoint_sha256')
        if ver.get('fidelity_claim') == 'changed_output':
            fail(errors, 'P8 AR-target-verification path cannot be labelled changed_output; use bounded_approximation or a separate P9 path')
    if p == 'P9':
        if ver.get('fidelity_claim') != 'changed_output':
            fail(errors, 'P9 must explicitly declare changed_output semantics')
        for key in ['draft_model_revision','draft_geometry','draft_checkpoint_sha256','draft_tokenizer_sha256','draft_config_sha256']:
            if ver.get(key) in (None, ''): fail(errors, f'P9 verification.{key} is required')
        if ver.get('draft_checkpoint_sha256') and data.get('model',{}).get('checkpoint_sha256') != ver.get('draft_checkpoint_sha256'):
            fail(errors, 'P9 AR seed/draft checkpoint must match model.checkpoint_sha256')
        required_p9_metrics={'correct_token_retention','edit_damage_rate','correction_precision','correction_recall','refinement_convergence'}
        present={m.get('name') for m in metrics if isinstance(m,dict)}
        if not required_p9_metrics.issubset(present):
            fail(errors, f'P9 evaluation metrics missing: {sorted(required_p9_metrics-present)}')
        ratio_metrics=required_p9_metrics
        for metric in metrics:
            if isinstance(metric,dict) and metric.get('name') in ratio_metrics:
                name=metric.get('name'); value=metric.get('value')
                numerator=metric.get('numerator'); denominator=metric.get('denominator')
                if metric.get('value_kind') != 'ratio':
                    fail(errors, f'P9 metric {name} must declare value_kind=ratio')
                if not isinstance(numerator,(int,float)) or isinstance(numerator,bool) or numerator < 0:
                    fail(errors, f'P9 metric {name} requires a non-negative numeric numerator')
                if not isinstance(denominator,(int,float)) or isinstance(denominator,bool) or denominator < 0:
                    fail(errors, f'P9 metric {name} requires a non-negative numeric denominator')
                if isinstance(numerator,(int,float)) and isinstance(denominator,(int,float)) and not isinstance(numerator,bool) and not isinstance(denominator,bool):
                    if denominator == 0:
                        if numerator != 0 or value != 'undefined':
                            fail(errors, f"P9 metric {name} with zero denominator must record numerator=0 and value='undefined'")
                    else:
                        expected=numerator/denominator
                        if not isinstance(value,(int,float)) or isinstance(value,bool):
                            fail(errors, f'P9 metric {name} must have a numeric value when denominator is non-zero')
                        elif value < 0 or value > 1:
                            fail(errors, f"P9 fraction metric {name} must be within [0,1]")
                        elif abs(value-expected) > max(1e-9,1e-6*abs(expected)):
                            fail(errors, f'P9 metric {name} value does not match numerator/denominator')
    if p == 'P7':
        serving = data.get('serving', {})
        if not serving.get('generation_modes') or not serving.get('mode_router_revision'):
            fail(errors, 'P7 requires generation_modes and mode_router_revision')
    names = [m.get('name') for m in metrics if isinstance(m, dict)]
    if len(names) != len(set(names)):
        fail(errors, 'evaluation metric names must be unique within a run')
    for metric in metrics:
        if not isinstance(metric,dict): continue
        if metric.get('value_kind') == 'ratio':
            numerator=metric.get('numerator'); denominator=metric.get('denominator'); value=metric.get('value')
            if isinstance(numerator,(int,float)) and not isinstance(numerator,bool) and isinstance(denominator,(int,float)) and not isinstance(denominator,bool):
                if denominator == 0:
                    if numerator != 0 or value != 'undefined':
                        fail(errors, f"ratio metric {metric.get('name')} with zero denominator must record numerator=0 and value='undefined'")
                elif not isinstance(value,(int,float)) or isinstance(value,bool) or abs(value-(numerator/denominator)) > max(1e-9,1e-6*abs(numerator/denominator)):
                    fail(errors, f"ratio metric {metric.get('name')} value must match numerator/denominator")
        if isinstance(metric.get('confidence_interval'),dict):
            ci=metric['confidence_interval']; lo=ci.get('lower'); hi=ci.get('upper')
            if isinstance(lo,(int,float)) and isinstance(hi,(int,float)) and lo > hi:
                fail(errors, f"metric {metric.get('name')} confidence interval lower bound exceeds upper bound")
    if data.get('run_type') in {'inference','evaluation','ablation','speculative_decode'}:
        serving = data.get('serving', {})
        if isinstance(serving, dict) and serving.get('measurement_iterations', 0) < 1:
            fail(errors, 'serving measurement_iterations must be >= 1')

def main():
    ap = argparse.ArgumentParser(description='Validate the LLM Architecture v9 research package and optional run manifest.')
    ap.add_argument('--root', type=Path, default=ROOT, help='extracted package root')
    ap.add_argument('--manifest', type=Path, help='optional concrete YAML/JSON run manifest to validate')
    args = ap.parse_args(); root = args.root.resolve(); errors=[]; checks=[]
    def check(label, condition, detail=''):
        if condition: checks.append(label)
        else: fail(errors, f'{label}: {detail or "failed"}')
    # Core documents and text mirror.
    md = root/'SOTA_LLM_Architecture_v9.md'; txt = root/'SOTA_LLM_Architecture_v9.txt'
    check('main specification exists', md.is_file(), str(md))
    check('TXT mirror exists', txt.is_file(), str(txt))
    if md.is_file() and txt.is_file():
        bmd, btxt = md.read_bytes(), txt.read_bytes()
        check('Markdown/TXT byte equality', bmd == btxt)
        doc = bmd.decode('utf-8')
        headers = [int(x) for x in re.findall(r'^## (\d+)\.', doc, re.M)]
        check(f'main sections sequential 0..{max(headers) if headers else -1}', bool(headers) and headers == list(range(max(headers) + 1)), str(headers))
        subs = re.findall(r'^### (\d+(?:\.\d+)+)\s', doc, re.M)
        dups = sorted({x for x in subs if subs.count(x)>1})
        check('no duplicate subsection identifiers', not dups, str(dups))
        check('Markdown fences balanced', len(re.findall(r'^```', doc, re.M)) % 2 == 0)
        definitions = set(re.findall(r'^- \*\*\[R(\d+)\]\*\*', doc, re.M))
        # Exclude bibliography body when computing use sites, but preserve citations in later appendices.
        a = doc.find('## 21. Research References')
        b = doc.find('\n## 22.', a+1) if a >= 0 else -1
        outside = doc if a < 0 or b < 0 else doc[:a] + doc[b:]
        uses = set()
        for group in re.findall(r'\[([^\]]*R\d+[^\]]*)\]', outside):
            for token in re.findall(r'R(\d+)', group):
                uses.add(token)
            for lo, hi in re.findall(r'R(\d+)\s*[–-]\s*R?(\d+)', group):
                start, end = int(lo), int(hi)
                if end >= start and end - start <= 1000:
                    uses.update(str(n) for n in range(start, end + 1))
        undefined = sorted(uses-definitions, key=int)
        unused = sorted(definitions-uses, key=int)
        check('all reference citations defined', not undefined, str(undefined))
        check('all bibliography entries cited', not unused, str(unused))
        max_reference = max((int(x) for x in definitions), default=0)
        check(f'reference IDs contiguous R1..R{max_reference}', bool(definitions) and definitions == {str(i) for i in range(1, max_reference + 1)}, f'{len(definitions)} definitions')
    # JSON schemas themselves.
    for schema_name in ['experiment_manifest.schema.json','candidate_profiles.schema.json','benchmark_contract.schema.json','kv_policy_matrix.schema.json','implementation_roadmap.schema.json','metric_definitions.schema.json']:
        p=root/schema_name
        try:
            sch=load_json(p); Draft202012Validator.check_schema(sch); check(f'{schema_name} parses and meta-validates',True)
        except Exception as e: fail(errors, f'{schema_name} invalid: {e}')
    # YAML profile / contract / cache matrix / roadmap.
    schema_pairs=[('candidate_profiles_v9.yaml','candidate_profiles.schema.json'),('benchmark_contract_v9.yaml','benchmark_contract.schema.json'),('kv_policy_matrix_v9.yaml','kv_policy_matrix.schema.json'),('implementation_roadmap_v9.yaml','implementation_roadmap.schema.json'),('metric_definitions_v9.yaml','metric_definitions.schema.json')]
    loaded={}
    for name,schname in schema_pairs:
        try:
            data=load_yaml(root/name); loaded[name]=data; schema=load_json(root/schname)
            validator=Draft202012Validator(schema, format_checker=FormatChecker())
            es=list(validator.iter_errors(data))
            check(f'{name} validates against schema', not es, '; '.join(e.message for e in es[:5]))
        except Exception as e: fail(errors,f'{name} parse/validation error: {e}')
    try:
        bc=loaded['benchmark_contract_v9.yaml']
        # Canonical paths must refer to fields represented by the normative run schema.
        allowed_paths={
          'schema_version','run_id','run_type','profile_id',
          'specification.revision','specification.file_name','specification.sha256',
          'source.repository','source.commit','source.dirty_tree','source.patch_sha256',
          'dataset[*].name','dataset[*].version','dataset[*].manifest_sha256','dataset[*].role','dataset[*].license_status','dataset[*].provenance_uri','dataset[*].allowed_uses','dataset[*].rights_review_status',
          'hardware.compute_device_type','hardware.compute_device_model','hardware.compute_device_count','hardware.device_memory_type','hardware.device_memory_per_device_gib','hardware.device_inventory_manifest_sha256','hardware.host_ram_gib','hardware.interconnect','hardware.topology','hardware.power_measurement',
          'software.framework','software.runtime','software.driver','software.compiler','software.environment_kind','software.environment_digest','software.dependency_lock_sha256','software.kernel_revisions',
          'model.total_parameters','model.active_parameters','model.tokenizer_revision','model.tokenizer_sha256','model.architecture_config_sha256','model.context_length','model.precision','model.quantization_map','model.checkpoint_sha256',
          'evaluation.benchmark_suite','evaluation.grader_revision','evaluation.sample_count','evaluation.repetitions','evaluation.confidence_interval_method','evaluation.metrics[*].name','evaluation.metrics[*].value','evaluation.metrics[*].value_kind','evaluation.metrics[*].unit','evaluation.metrics[*].protocol','evaluation.metrics[*].numerator','evaluation.metrics[*].denominator',
          'artifacts.configuration_sha256','artifacts.raw_logs_sha256',
          'training.optimizer','training.optimizer_config','training.scheduler','training.learning_rate_schedule','training.global_batch_tokens','training.training_tokens','training.seed','training.precision','training.max_sequence_length','training.gradient_checkpointing','training.parent_checkpoint_sha256','training.reward_model_revision','training.reward_data_manifest_sha256',
          'serving.batch_size','serving.concurrency','serving.prompt_length_distribution','serving.output_length_distribution','serving.decoding_settings','serving.cache_policy_revision','serving.generation_mode','serving.warmup_iterations','serving.measurement_iterations','serving.latency_definition','serving.generation_modes','serving.mode_router_revision',
          'verification.draft_model_revision','verification.draft_geometry','verification.draft_checkpoint_sha256','verification.draft_tokenizer_sha256','verification.draft_config_sha256','verification.approximation_metric','verification.approximation_bound','verification.verifier_algorithm_revision','verification.state_commit_policy_revision','verification.proposal_probability_available','verification.target_checkpoint_sha256','verification.acceptance_correction_rule','verification.fidelity_test_revision','verification.fidelity_test_result','verification.fidelity_claim'
        }
        path_values=bc['required_environment_fields']+bc['required_run_fields']
        path_values += [v for group in bc['conditional_run_fields'].values() for v in group]
        # Dataset role predicates and metric-index selectors are explicit contract expressions, not schema object paths.
        special_expressions={
          'dataset[role=training].allowed_uses=training',
          'dataset[role=training].rights_review_status=approved_for_declared_uses',
          'dataset[role=reward].allowed_uses=reward_model_training',
          'dataset[role=reward].rights_review_status=approved_for_declared_uses',
          'evaluation.metrics[correct_token_retention]','evaluation.metrics[edit_damage_rate]',
          'evaluation.metrics[correction_precision]','evaluation.metrics[correction_recall]',
          'evaluation.metrics[refinement_convergence]',
          'verification.fidelity_claim=changed_output',
          'evaluation.metrics[correct_token_retention].value_kind=ratio','evaluation.metrics[correct_token_retention].numerator','evaluation.metrics[correct_token_retention].denominator',
          'evaluation.metrics[edit_damage_rate].value_kind=ratio','evaluation.metrics[edit_damage_rate].numerator','evaluation.metrics[edit_damage_rate].denominator',
          'evaluation.metrics[correction_precision].value_kind=ratio','evaluation.metrics[correction_precision].numerator','evaluation.metrics[correction_precision].denominator',
          'evaluation.metrics[correction_recall].value_kind=ratio','evaluation.metrics[correction_recall].numerator','evaluation.metrics[correction_recall].denominator',
          'evaluation.metrics[refinement_convergence].value_kind=ratio','evaluation.metrics[refinement_convergence].numerator','evaluation.metrics[refinement_convergence].denominator'
        }
        invalid=[path_value for path_value in path_values if path_value not in allowed_paths and path_value not in special_expressions]
        check('benchmark field paths align with run schema',not invalid,str(invalid[:12]))
        check('benchmark environment fields are included in common run identity',set(bc['required_environment_fields']).issubset(set(bc['required_run_fields'])),str(sorted(set(bc['required_environment_fields'])-set(bc['required_run_fields']))))
        expected_role_uses={'training':['training'],'validation':['validation','model_selection'],'test':['evaluation'],'benchmark':['evaluation'],'reward':['reward_model_training']}
        check('dataset role/use contract exactly matches manifest validator policy',bc['dataset_role_use_requirements']==expected_role_uses,str(bc['dataset_role_use_requirements']))
        expected_condition_groups={'pretraining','continued_pretraining_sft_rl','rl','inference_evaluation_ablation_speculative_decode','P7','P8','P9','dirty_source_tree','metric_value_kind_ratio'}
        check('conditional field groups are complete',set(bc['conditional_run_fields'])==expected_condition_groups,str(sorted(bc['conditional_run_fields'])))
        check('dirty-source patch digest is conditional, not universal',bc['conditional_run_fields'].get('dirty_source_tree')==['source.patch_sha256'],'source.patch_sha256 must be conditional on dirty_tree=true')
        check('benchmark run fields do not use obsolete aliases',not ({'dataset_manifest_hash','hardware_manifest_hash','software_lockfile_hash','model_revision','tokenizer_revision','accelerator_model','accelerator_count','HBM_per_device'} & set(bc['required_run_fields']+bc['required_environment_fields'])), 'obsolete alias remains')
    except Exception as e: fail(errors,f'benchmark path/semantic check: {e}')
    try:
        c=loaded['candidate_profiles_v9.yaml']; ids=[p['id'] for p in c['profiles']]
        check('candidate profile IDs unique',len(ids)==len(set(ids)),str(ids))
        check('P0..P9 candidate inventory complete',set(ids)=={'P0','P1','P1-LA','P2','P3','P4','P5','P6','P7','P8','P9'},str(ids))
    except Exception as e: fail(errors,f'candidate inventory semantic check: {e}')
    try:
        k=loaded['kv_policy_matrix_v9.yaml']; pids=[p['id'] for p in k['policies']]
        check('KV policy IDs unique',len(pids)==len(set(pids)),str(pids))
        check('full KV cache is declared baseline',k['baseline']=='full_KV_cache',str(k['baseline']))
        check('KV policy composition is one-at-a-time by default',k['composition_policy']['default']=='one_candidate_at_a_time',str(k['composition_policy']))
    except Exception as e: fail(errors,f'KV matrix semantic check: {e}')
    try:
        road=loaded['implementation_roadmap_v9.yaml']
        stage_ids=[s['id'] for s in road['stages']]
        check('roadmap gate IDs unique',len(stage_ids)==len(set(stage_ids)),str(stage_ids))
        check('roadmap starts with envelope then P0',stage_ids[:2]==['G0','G1'],str(stage_ids[:2]))
        known=set(stage_ids); positions={sid:i for i,sid in enumerate(stage_ids)}; bad=[]
        for stage in road['stages']:
            deps=stage.get('depends_on',[])+stage.get('optional_dependencies',[])
            for dep in deps:
                if dep not in known or positions.get(dep,10**9)>=positions[stage['id']]: bad.append(f"{stage['id']}->{dep}")
        check('roadmap dependencies exist and point backward',not bad,', '.join(bad))
    except Exception as e: fail(errors,f'roadmap YAML error: {e}')
    try:
        defs=loaded['metric_definitions_v9.yaml']['metrics']; mids=[m['id'] for m in defs]
        check('metric definition IDs unique',len(mids)==len(set(mids)),str(mids))
        required_metric_ids={'quality_at_fixed_compute','cost_per_accepted_token','draft_acceptance_by_position','state_realization_equivalence','correct_token_retention','edit_damage_rate','correction_precision','correction_recall','refinement_convergence','time_to_first_token','inter_token_latency','end_to_end_latency','aggregate_throughput','energy_per_task'}
        check('core metric definitions complete',required_metric_ids.issubset(set(mids)),str(sorted(required_metric_ids-set(mids))))
    except Exception as e: fail(errors,f'metric definitions YAML error: {e}')
    try:
        templ=load_yaml(root/'experiment_manifest.template.yaml')
        check('incomplete manifest template is parseable YAML',isinstance(templ,dict))
        check('template visibly marked incomplete','REPLACE_ME' in (root/'experiment_manifest.template.yaml').read_text(encoding='utf-8'))
        sch=load_json(root/'experiment_manifest.schema.json')
        errs=list(Draft202012Validator(sch,format_checker=FormatChecker()).iter_errors(templ))
        check('incomplete template is rejected by run schema',bool(errs),'template unexpectedly passed')
    except Exception as e: fail(errors,f'manifest template check: {e}')
    # Checksums, if present.
    sums=root/'SHA256SUMS_v9.txt'
    if sums.is_file():
        rows=[ln.strip() for ln in sums.read_text(encoding='utf-8').splitlines() if ln.strip()]
        bad=[]
        for row in rows:
            try:
                digest, rel=row.split('  ',1); f=(root/rel)
                if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest: bad.append(rel)
            except Exception: bad.append(row)
        check('SHA-256 manifest entries verify',not bad,', '.join(bad[:10]))
        listed=[row.split('  ',1)[1] for row in rows if '  ' in row]
        root_assets={
          'CHANGELOG_v8_to_v9.md','LLM_Architecture_Audit_v8_to_v9.md','LLM_Architecture_Audit_v8_to_v9.txt',
          'README_LLM_Architecture_v9.md','SOTA_LLM_Architecture_v9.md','SOTA_LLM_Architecture_v9.txt',
          'TRAINING_PILOT_REPORT_v9.md','VALIDATION_REPORT_v9.md','benchmark_contract.schema.json','benchmark_contract_v9.yaml',
          'candidate_profiles.schema.json','candidate_profiles_v9.yaml','dataset_requirements.txt','experiment_manifest.schema.json',
          'experiment_manifest.template.yaml','implementation_roadmap.schema.json','implementation_roadmap_v9.yaml',
          'kv_policy_matrix.schema.json','kv_policy_matrix_v9.yaml','metric_definitions.schema.json',
          'metric_definitions_v9.yaml','merge_pilot_results.py','prepare_hf_dataset.py','pilot_requirements.txt','requirements.txt',
          'test_dataset_pipeline.py','test_validate_architecture_package.py','tinyshakespeare_30k.txt',
          'train_pilot.py','validate_architecture_package.py','verify_pilot_artifacts.py',
          'results/P0_seed20261118.pt','results/P0_seed20261119.pt','results/P0_seed903341.pt','results/P1_seed20261118.pt',
          'results/P1_seed20261119.pt','results/P1-LA_seed20261118.pt','results/P1-LA_seed20261119.pt',
          'results/dataset_manifest.json','results/pilot_results.json','results/pilot_metrics.csv','results/pilot_aggregate.csv','results/pilot_seed_20261118_current.json',
          'results/pilot_seed_20261119_current.json','results/repeat_20261118_current.json','results/source_locked_repeat_20261118.json',
          'results/pilot_verification_report.json','results/bundle_loader_hardened_smoke.json',
          'results/bundle_loader_hardened_smoke.log','results/seed20261118_current.log',
          'results/repeat_20261118_current.log',
          'results/local_split_bundle_smoke/train.txt','results/local_split_bundle_smoke/validation.txt',
          'results/local_split_bundle_smoke/test.txt','results/local_split_bundle_smoke/dataset_manifest.json',
          'results/local_split_bundle_smoke/dataset_manifest.sha256'
        }
        expected=[]
        for f in root.rglob('*'):
            if not f.is_file() or '__pycache__' in f.parts or f.suffix == '.pyc': continue
            rel=f.relative_to(root).as_posix()
            if rel in root_assets or rel.startswith('history/') or rel.startswith('inputs/'):
                if rel not in {'SHA256SUMS_v9.txt','history/v8/build_v8.py'}: expected.append(rel)
        expected=sorted(expected)
        check('distributable root asset inventory is complete',all((root/name).is_file() for name in root_assets),str(sorted(name for name in root_assets if not (root/name).is_file())))
        check('SHA-256 manifest covers every package file exactly once',len(listed)==len(set(listed)) and set(listed)==set(expected),f'missing={sorted(set(expected)-set(listed))[:5]} extra={sorted(set(listed)-set(expected))[:5]}')
    # Included source archives.
    for name in ['inputs/ideas20_full_compilation.zip','inputs/LLM_Architecture_v3_Research_Package.zip','inputs/LLM_Architecture_v7_Research_Package.zip','inputs/LLM_Architecture_v8_Research_Package.zip']:
        p=root/name
        if p.is_file():
            try:
                with zipfile.ZipFile(p) as z: bad=z.testzip()
                check(f'{name} ZIP integrity',bad is None,str(bad))
            except Exception as e: fail(errors,f'{name} archive error: {e}')
    # Optional real run manifest.
    if args.manifest:
        mp=args.manifest.resolve()
        try:
            data=json.loads(mp.read_text(encoding='utf-8')) if mp.suffix.lower()=='.json' else yaml.safe_load(mp.read_text(encoding='utf-8'))
            sch=load_json(root/'experiment_manifest.schema.json')
            es=list(Draft202012Validator(sch,format_checker=FormatChecker()).iter_errors(data))
            check('supplied run manifest passes JSON Schema',not es,'; '.join(e.message for e in es[:10]))
            semantic=[]; semantic_manifest_checks(data,semantic)
            check('supplied run manifest passes semantic checks',not semantic,'; '.join(semantic))
        except Exception as e: fail(errors,f'supplied manifest could not be checked: {e}')
    print(f'LLM Architecture v9 package validator — {len(checks)} checks passed; {len(errors)} errors')
    for item in checks: print(f'PASS  {item}')
    for item in errors: print(f'FAIL  {item}')
    if errors: return 1
    print('NOTE  Static/preflight validation only. This does not validate training, kernels, model quality, licensing decisions, or scientific claims.')
    return 0

if __name__=='__main__': sys.exit(main())
