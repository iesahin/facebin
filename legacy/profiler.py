import cProfile
import datetime
import pstats
import main_gui

filename = datetime.datetime.now().strftime(
    "/tmp/facebin-profile-result-%F-%H-%M-%S-%f.prof")
cProfile.run('main_gui.main()', filename=filename)
p = pstats.Stats(filename)
p.sort_stats('cumulative').print_stats(100)
