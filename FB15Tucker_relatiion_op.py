from pykeen.datasets.freebase import FB15k237
import torch
dataset = FB15k237(create_inverse_triples=True)
training_triples_factory = dataset.training

# Pick a model
from pykeen.models import TuckER
model = TuckER(triples_factory=training_triples_factory)
model.to('cuda')

# Pick an optimizer from Torch
from torch.optim import Adam
optimizer = Adam(params=model.get_grad_params())

# Pick a training approach (sLCWA or LCWA)
from pykeen.training import LCWATrainingLoop
training_loop = LCWATrainingLoop(
    target=1,
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer)
  #  callbacks=StopperTrainingCallback(stopper=stopper_1,triples_factory=training_triples_factory))

# Train like Cristiano Ronaldo
_ = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=100,
    batch_size=256)

torch.save(model,"Wiki5M/Tuker_relation_100_v0")
