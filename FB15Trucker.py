from pykeen.hpo import hpo_pipeline
hpo_pipeline_result = hpo_pipeline(
    n_trials=30,
    dataset='FB15k-237',
    model='TuckER',
    evaluator_kwargs=dict(filtered=True),
    stopper='early',
    stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
)

hpo_pipeline_result.save_to_directory('fb15trucker')
