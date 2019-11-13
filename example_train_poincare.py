import pickle
import config
import models
import json
import os 
import numpy as np

con = config.Config()
#Input training files from benchmarks/FB15K237/ folder.
con.set_in_path("./benchmarks/FB15K237/") # WN18RR
con.set_gpu(True)
# con.set_int_type('int64')
#True: Input test files from the same folder.
con.set_test_link_prediction(True)
con.set_log_on(1)
con.set_work_threads(8)
con.set_train_times(2000)
con.set_nbatches(10)
con.set_alpha(0.01)
con.set_bern(1)
con.set_dimension(100)
con.set_margin(0.5)
con.set_lmbda(0.2)
con.set_ent_neg_rate(5)
con.set_rel_neg_rate(0)
con.set_opt_method("RSGD")
#Model parameters will be exported via torch.save() automatically.
con.set_export_files("./res/hyperkg.pt")
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/hyperkg_embedding.vec.json")
con.init()
con.set_model(models.Poincare)
con.run()

embeddings = con.get_parameters("numpy")

ent_embeddings = embeddings["ent_embeddings.weight"]
rel_embeddings = embeddings["rel_embeddings.weight"]
assert not np.isnan(ent_embeddings).any()
assert not np.isnan(rel_embeddings).any()

norms_ent_embeddings = np.linalg.norm(ent_embeddings, axis=1)
norms_rel_embeddings = np.linalg.norm(rel_embeddings, axis=1)
np.testing.assert_array_less(norms_ent_embeddings, 0.5)
np.testing.assert_array_less(norms_rel_embeddings, 1.0)