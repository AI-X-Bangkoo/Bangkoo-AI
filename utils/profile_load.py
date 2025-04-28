# profile_load.py
import cProfile, pstats, io
from model_loader import model_manager

pr = cProfile.Profile()
pr.enable()

model_manager.load()

pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
ps.print_stats(10)

print(s.getvalue())
