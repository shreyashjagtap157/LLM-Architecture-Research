#!/usr/bin/env python3
"""Regression tests for the LLM Architecture v9 manifest contract."""
import importlib.util, json, unittest
from pathlib import Path
import yaml
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('package_validator', ROOT/'validate_architecture_package.py')
validator=importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
SCHEMA=json.loads((ROOT/'experiment_manifest.schema.json').read_text())

class ManifestContractTests(unittest.TestCase):
    def setUp(self):
        self.data={
          'schema_version':'1.0','run_id':'test-p0-eval-001','run_type':'evaluation','profile_id':'P0',
          'specification':{'revision':'v9','file_name':'SOTA_LLM_Architecture_v9.md','sha256':'a'*64},
          'source':{'repository':'urn:repo:test','commit':'a'*40,'dirty_tree':False},
          'dataset':[{'name':'test-data','version':'1','manifest_sha256':'b'*64,'role':'test','license_status':'evaluation_only','provenance_uri':'urn:dataset:test','allowed_uses':['evaluation'],'rights_review_status':'approved_for_declared_uses'}],
          'hardware':{'compute_device_type':'accelerator_only','compute_device_model':'fixture','compute_device_count':1,'device_memory_type':'HBM','device_memory_per_device_gib':80,'device_inventory_manifest_sha256':'b'*64,'host_ram_gib':256,'interconnect':'fixture-link','topology':'single-node','power_measurement':'unavailable'},
          'software':{'framework':'fixture','runtime':'fixture','driver':'n/a','compiler':'n/a','environment_kind':'bare_metal','environment_digest':'c'*64,'kernel_revisions':['reference-v1'],'dependency_lock_sha256':'c'*64},
          'model':{'total_parameters':1000000,'active_parameters':1000000,'tokenizer_revision':'tok-v1','tokenizer_sha256':'b'*64,'architecture_config_sha256':'a'*64,'context_length':4096,'precision':'fp32','quantization_map':'none','checkpoint_sha256':'d'*64},
          'serving':{'batch_size':1,'concurrency':1,'prompt_length_distribution':'fixed 512','output_length_distribution':'fixed 128','decoding_settings':{'temperature':0},'cache_policy_revision':'full-cache-v1','generation_mode':'autoregressive','warmup_iterations':1,'measurement_iterations':3,'latency_definition':'end_to_end'},
          'evaluation':{'benchmark_suite':['synthetic-fixture'],'grader_revision':'fixture-v1','sample_count':3,'repetitions':1,'confidence_interval_method':'not_estimated_fixture','metrics':[{'name':'exact_match','value':1.0,'value_kind':'ratio','numerator':1,'denominator':1,'unit':'fraction','protocol':'synthetic schema fixture only'}]},
          'artifacts':{'configuration_sha256':'e'*64,'raw_logs_sha256':'f'*64}
        }
    def errors(self, data):
        schema_errors=list(Draft202012Validator(SCHEMA).iter_errors(data))
        semantic=[]; validator.semantic_manifest_checks(data,semantic)
        return schema_errors+semantic
    def test_valid_p0_evaluation(self): self.assertEqual([],self.errors(self.data))
    def test_active_parameters_cannot_exceed_total(self):
        d=json.loads(json.dumps(self.data)); d['model']['active_parameters']=1000001
        self.assertTrue(any('active_parameters' in str(e) for e in self.errors(d)))
    def test_dirty_tree_requires_patch_digest(self):
        d=json.loads(json.dumps(self.data)); d['source']['dirty_tree']=True
        self.assertTrue(self.errors(d))
    def test_p8_exact_requires_probability_and_passing_test(self):
        d=json.loads(json.dumps(self.data)); d.update(run_type='speculative_decode',profile_id='P8')
        d['verification']={'fidelity_claim':'exact_target_distribution','draft_model_revision':'draft-v1','draft_geometry':'causal_path_from_token_lattice','verifier_algorithm_revision':'verify-v1','state_commit_policy_revision':'commit-v1','proposal_probability_available':True,'acceptance_correction_rule':'exact-correction-v1','target_checkpoint_sha256':'d'*64,'draft_checkpoint_sha256':'8'*64,'draft_tokenizer_sha256':'7'*64,'draft_config_sha256':'6'*64,'fidelity_test_revision':'fidelity-v1','fidelity_test_result':'passed'}
        self.assertEqual([],self.errors(d))
        d['verification']['proposal_probability_available']=False
        self.assertTrue(self.errors(d))
    def test_p8_bounded_approximation_requires_metric_and_bound(self):
        d=json.loads(json.dumps(self.data)); d.update(run_type='speculative_decode',profile_id='P8')
        d['verification']={'fidelity_claim':'bounded_approximation','draft_model_revision':'draft-v1','draft_geometry':'causal-path','draft_checkpoint_sha256':'8'*64,'draft_tokenizer_sha256':'7'*64,'draft_config_sha256':'6'*64,'verifier_algorithm_revision':'verify-v1','state_commit_policy_revision':'commit-v1','proposal_probability_available':False,'target_checkpoint_sha256':'d'*64,'acceptance_correction_rule':'approx-v1','fidelity_test_revision':'fidelity-v1','fidelity_test_result':'passed','approximation_metric':'total_variation_distance','approximation_bound':0.02}
        d['evaluation']['metrics'].append({'name':'total_variation_distance','value':0.01,'value_kind':'scalar','unit':'fraction','protocol':'synthetic test'})
        self.assertEqual([],self.errors(d))
        d['verification'].pop('approximation_bound')
        self.assertTrue(self.errors(d))
        d['verification']['approximation_bound']=0.005
        self.assertTrue(any('exceeds declared approximation_bound' in str(e) for e in self.errors(d)))

    def test_rl_requires_reward_dataset(self):
        d=json.loads(json.dumps(self.data)); d.update(run_type='rl'); d.pop('serving')
        d['training']={'optimizer':'AdamW','optimizer_config':{'learning_rate':1e-5},'scheduler':'cosine','learning_rate_schedule':'warmup-cosine','global_batch_tokens':1000,'training_tokens':10000,'seed':2,'precision':'bf16','max_sequence_length':4096,'gradient_checkpointing':True,'parent_checkpoint_sha256':'d'*64}
        self.assertTrue(any('requires at least one dataset with role=reward' in str(e) for e in self.errors(d)))
        d['dataset'].append({'name':'reward-data','version':'1','manifest_sha256':'c'*64,'role':'reward','license_status':'evaluation_only','provenance_uri':'urn:dataset:reward','allowed_uses':['reward_model_training'],'rights_review_status':'approved_for_declared_uses'})
        self.assertFalse(any('requires at least one dataset with role=reward' in str(e) for e in self.errors(d)))

    def test_evaluation_requires_evaluation_dataset(self):
        d=json.loads(json.dumps(self.data)); d['dataset'][0]['allowed_uses']=['training']
        self.assertTrue(any('requires a test/benchmark dataset approved for evaluation' in str(e) for e in self.errors(d)))

    def test_p9_zero_denominator_is_undefined_not_zero(self):
        d=json.loads(json.dumps(self.data)); d.update(run_type='speculative_decode',profile_id='P9')
        d['verification']={'fidelity_claim':'changed_output','draft_model_revision':'ar-v1','draft_geometry':'masked-refinement','draft_checkpoint_sha256':'d'*64,'draft_tokenizer_sha256':'b'*64,'draft_config_sha256':'c'*64}
        for name in ['correct_token_retention','edit_damage_rate','correction_precision','correction_recall','refinement_convergence']:
            d['evaluation']['metrics'].append({'name':name,'value':0.5,'value_kind':'ratio','numerator':1,'denominator':2,'unit':'fraction','protocol':'synthetic fixture'})
        metric=next(m for m in d['evaluation']['metrics'] if m['name']=='edit_damage_rate')
        metric.update(value='undefined',numerator=0,denominator=0)
        self.assertEqual([],self.errors(d))
        metric['value']=0.0
        self.assertTrue(any("value='undefined'" in str(e) for e in self.errors(d)))

    def test_p8_failed_fidelity_can_be_recorded_as_unverified(self):
        d=json.loads(json.dumps(self.data)); d.update(run_type='speculative_decode',profile_id='P8')
        d['verification']={'fidelity_claim':'unverified','draft_model_revision':'draft-v1','draft_geometry':'causal-path','draft_checkpoint_sha256':'8'*64,'draft_tokenizer_sha256':'7'*64,'draft_config_sha256':'6'*64,'verifier_algorithm_revision':'verify-v1','state_commit_policy_revision':'commit-v1','proposal_probability_available':False,'target_checkpoint_sha256':'d'*64,'acceptance_correction_rule':'not-established','fidelity_test_revision':'fidelity-v1','fidelity_test_result':'failed'}
        self.assertEqual([],self.errors(d))

    def test_p9_requires_changed_output(self):
        d=json.loads(json.dumps(self.data)); d.update(run_type='speculative_decode',profile_id='P9')
        d['verification']={'fidelity_claim':'changed_output','draft_model_revision':'ar-draft-v1','draft_geometry':'masked-refinement','draft_checkpoint_sha256':'d'*64,'draft_tokenizer_sha256':'b'*64,'draft_config_sha256':'c'*64}
        for name in ['correct_token_retention','edit_damage_rate','correction_precision','correction_recall','refinement_convergence']:
            d['evaluation']['metrics'].append({'name':name,'value':0.5,'value_kind':'ratio','numerator':1,'denominator':2,'unit':'fraction','protocol':'synthetic fixture'})
        self.assertEqual([],self.errors(d))
        d['verification']['fidelity_claim']='exact_target_distribution'
        self.assertTrue(self.errors(d))
    def test_p7_requires_mode_router(self):
        d=json.loads(json.dumps(self.data)); d['profile_id']='P7'
        d['serving'].pop('mode_router_revision',None); d['serving'].pop('generation_modes',None)
        self.assertTrue(self.errors(d))
    def test_template_is_not_valid_run_manifest(self):
        template=yaml.safe_load((ROOT/'experiment_manifest.template.yaml').read_text())
        self.assertTrue(list(Draft202012Validator(SCHEMA).iter_errors(template)))
    def test_generic_ratio_value_must_match_raw_counts(self):
        d=json.loads(json.dumps(self.data))
        d['evaluation']['metrics'][0]['value']=0.5
        self.assertTrue(any('ratio metric exact_match value must match numerator/denominator' in str(e) for e in self.errors(d)))

    def test_confidence_interval_bounds_must_be_ordered(self):
        d=json.loads(json.dumps(self.data))
        d['evaluation']['metrics'][0]['confidence_interval']={'lower':0.8,'upper':0.6,'confidence_level':0.95,'method':'bootstrap'}
        self.assertTrue(any('lower bound exceeds upper bound' in str(e) for e in self.errors(d)))

    def test_duplicate_metric_names_rejected(self):
        d=json.loads(json.dumps(self.data)); d['evaluation']['metrics'].append(dict(d['evaluation']['metrics'][0]))
        self.assertTrue(any('metric names must be unique' in str(e) for e in self.errors(d)))
    def test_nested_placeholder_rejected(self):
        d=json.loads(json.dumps(self.data)); d['serving']['cache_policy_revision']='REPLACE_ME'
        self.assertTrue(any('placeholder value remains' in str(e) for e in self.errors(d)))

    def test_cpu_only_bare_metal_reference_is_supported(self):
        d=json.loads(json.dumps(self.data)); d['hardware']={'compute_device_type':'cpu_only','compute_device_model':'x86_64-reference-host','compute_device_count':1,'device_memory_type':'system_ram','device_memory_per_device_gib':None,'device_inventory_manifest_sha256':'b'*64,'host_ram_gib':64,'interconnect':'not_applicable','topology':'single-host','power_measurement':'unavailable'}
        d['software']={'framework':'PyTorch','runtime':'eager','driver':'not_applicable','compiler':'not_applicable','environment_kind':'bare_metal','environment_digest':'c'*64,'kernel_revisions':['reference-kernel'],'dependency_lock_sha256':'c'*64}
        self.assertEqual([],self.errors(d))
    def test_training_role_requires_explicit_use_and_approval(self):
        d=json.loads(json.dumps(self.data)); d.update(run_type='pretraining'); d.pop('serving')
        d['dataset']=[{'name':'train-data','version':'1','manifest_sha256':'b'*64,'role':'training','license_status':'unknown','provenance_uri':'urn:dataset:train','allowed_uses':['evaluation'],'rights_review_status':'pending'}]
        d['training']={'optimizer':'AdamW','optimizer_config':{'learning_rate':1e-4},'scheduler':'cosine','learning_rate_schedule':'warmup-cosine','global_batch_tokens':100000,'training_tokens':1000000,'seed':7,'precision':'bf16','max_sequence_length':4096,'gradient_checkpointing':True}
        errors=self.errors(d)
        self.assertTrue(any('lacks allowed_uses=training' in str(e) for e in errors))
        self.assertTrue(any('rights review is not approved' in str(e) for e in errors))

    def test_evaluation_dataset_requires_evaluation_rights(self):
        d=json.loads(json.dumps(self.data)); d['dataset'][0]['allowed_uses']=['training']
        self.assertTrue(any('lacks allowed_uses=evaluation' in str(e) for e in self.errors(d)))
        d['dataset'][0]['allowed_uses']=['evaluation']; d['dataset'][0]['rights_review_status']='pending'
        self.assertTrue(any('rights review is not approved' in str(e) for e in self.errors(d)))

    def test_validation_dataset_requires_model_selection_use(self):
        d=json.loads(json.dumps(self.data)); d['dataset'][0]['role']='validation'; d['dataset'][0]['allowed_uses']=['validation','evaluation']
        self.assertTrue(any('lacks model_selection use approval' in str(e) for e in self.errors(d)))
    def test_container_environment_requires_digest(self):
        d=json.loads(json.dumps(self.data)); d['software']['environment_kind']='container'
        self.assertTrue(self.errors(d))
        d['software']['container_digest']='sha256:'+'a'*64
        self.assertEqual([],self.errors(d))
    def test_heterogeneous_hardware_uses_inventory_manifest(self):
        d=json.loads(json.dumps(self.data)); d['hardware'].update(compute_device_type='heterogeneous',device_memory_type='mixed',device_memory_per_device_gib=None)
        self.assertEqual([],self.errors(d))

    def test_p8_target_checkpoint_must_match_model(self):
        d=json.loads(json.dumps(self.data)); d.update(run_type='speculative_decode',profile_id='P8')
        d['verification']={'fidelity_claim':'bounded_approximation','draft_model_revision':'draft-v1','draft_geometry':'causal_path_from_token_lattice','draft_checkpoint_sha256':'8'*64,'draft_tokenizer_sha256':'7'*64,'draft_config_sha256':'6'*64,'verifier_algorithm_revision':'verify-v1','state_commit_policy_revision':'commit-v1','proposal_probability_available':False,'target_checkpoint_sha256':'9'*64,'acceptance_correction_rule':'bounded-correction-v1','fidelity_test_revision':'test-v1','fidelity_test_result':'passed'}
        self.assertTrue(any('target checkpoint must match' in str(e) for e in self.errors(d)))
    def test_p9_requires_edit_quality_metrics(self):
        d=json.loads(json.dumps(self.data)); d.update(run_type='speculative_decode',profile_id='P9')
        d['verification']={'fidelity_claim':'changed_output','draft_model_revision':'ar-v1','draft_geometry':'masked-refinement','draft_checkpoint_sha256':'d'*64,'draft_tokenizer_sha256':'b'*64,'draft_config_sha256':'c'*64}
        self.assertTrue(any('P9 evaluation metrics missing' in str(e) for e in self.errors(d)))

    def test_p9_fraction_metrics_must_be_numeric_and_bounded(self):
        d=json.loads(json.dumps(self.data)); d.update(run_type='speculative_decode',profile_id='P9')
        d['verification']={'fidelity_claim':'changed_output','draft_model_revision':'ar-v1','draft_geometry':'masked-refinement','draft_checkpoint_sha256':'d'*64,'draft_tokenizer_sha256':'b'*64,'draft_config_sha256':'c'*64}
        for name in ['correct_token_retention','edit_damage_rate','correction_precision','correction_recall','refinement_convergence']:
            d['evaluation']['metrics'].append({'name':name,'value':0.5,'value_kind':'ratio','numerator':1,'denominator':2,'unit':'fraction','protocol':'synthetic fixture'})
        self.assertEqual([],self.errors(d))
        d['evaluation']['metrics'][-1]['value']=1.2; d['evaluation']['metrics'][-1]['numerator']=6; d['evaluation']['metrics'][-1]['denominator']=5
        self.assertTrue(any('must be within [0,1]' in str(e) for e in self.errors(d)))

    def test_p8_profile_requires_speculative_decode_run_type(self):
        d=json.loads(json.dumps(self.data)); d['profile_id']='P8'
        self.assertTrue(self.errors(d))

    def test_p9_draft_checkpoint_must_match_ar_model(self):
        d=json.loads(json.dumps(self.data)); d.update(run_type='speculative_decode',profile_id='P9')
        d['verification']={'fidelity_claim':'changed_output','draft_model_revision':'ar-v1','draft_geometry':'masked-refinement','draft_checkpoint_sha256':'9'*64,'draft_tokenizer_sha256':'b'*64,'draft_config_sha256':'c'*64}
        self.assertTrue(any('AR seed/draft checkpoint must match' in str(e) for e in self.errors(d)))

if __name__=='__main__': unittest.main(verbosity=2)
