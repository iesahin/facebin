import cProfile
import datetime
import pstats
import video_recorder

filename = datetime.datetime.now().strftime(
    "/tmp/video-recorder-profile-result-%F-%H-%M-%S-%f.prof")
cProfile.run('video_recorder.main()', filename=filename)
p = pstats.Stats(filename)
p.sort_stats('cumulative').print_stats(100)
