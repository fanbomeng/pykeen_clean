from pykeen.pipeline import pipeline

pipeline_result = pipeline(
    model='TuckER',
    dataset='FB15k-237',
   # result_tracker='tensorboard',
    training_loop='sLCWA',
    stopper='early',
    stopper_kwargs=dict(frequency=5, patience=2),
    epochs=20, 
    negative_sampler='basic',
    negative_sampler_kwargs=dict(
        corruption_scheme=('h', 'r', 't'),
    ),
    dataset_kwargs=dict(
        create_inverse_triples=True,
    ))
