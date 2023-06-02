# /home/ubuntu/anaconda3/envs/fanbo_pykeen/lib/python3.9/site-packages/pykeen/training/training_loop.py 
from pykeen.pipeline import pipeline
pipeline_result = pipeline(
    model='TuckER',
    dataset='FB15k-237',
    result_tracker='tensorboard',
    training_loop='LCWA',
    stopper='early',
    stopper_kwargs=dict(frequency=10, patience=5, relative_delta=0.002),
    epochs=100,
    dataset_kwargs=dict(
        create_inverse_triples=True,
    ))
pipeline_result.save_to_directory('FB15_100_epoch_base')
