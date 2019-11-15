import config
import models
import numpy as np
import json
import os

#Set import files and OpenKE will automatically load models via torch.load().
con = config.Config()
con.set_in_path("./benchmarks/FB15K237/") # FB15K237 , WN18RR
con.set_test_link_prediction(True)
con.set_log_on(1)
con.set_gpu(True)
con.set_int_type('int64')
con.set_work_threads(8)
con.set_dimension(100)
con.set_import_files("./path/to/the/hyperkg_saved_model.pt")
con.init()
con.set_model(models.Poincare)
con.test(-1,0)