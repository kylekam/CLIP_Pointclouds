from line_profiler import LineProfiler

from reconstruct import main, CLIP

lp = LineProfiler()


# main = lp(main)
# main()
# lp.print_stats()


lp.add_function(main)
lp.add_function(CLIP.getPatches)
wrapper = lp(main)
wrapper()

lp.print_stats()