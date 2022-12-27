import tracemalloc
 
# list to store memory snapshots
snaps = []
 
def snapshot():
	snaps.append(tracemalloc.take_snapshot())
 
 
def display_stats():
	stats = snaps[0].statistics('filename')
	print("\n*** top 5 stats grouped by filename ***")
	for s in stats[:5]:
    	    print(s)
 
 
def compare():
	first = snaps[0]
	for snapshot in snaps[1:]:
    	    stats = snapshot.compare_to(first, 'lineno')
    	    print("\n*** top 10 stats ***")
    	    for s in stats[:10]:
              print(s)
 
 
def print_trace():
	# pick the last saved snapshot, filter noise
	snapshot = snaps[-1].filter_traces((
    	    tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
    	    tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
    	    tracemalloc.Filter(False, "<unknown>"),
	))
	largest = snapshot.statistics("traceback")[0]
 
	print(f"\n*** Trace for largest memory block - ({largest.count} blocks, {largest.size/1024} Kb) ***")
	for l in largest.traceback.format():
    	    print(l)

	largest = snapshot.statistics("traceback")[1]
 
	print(f"\n*** Trace for largest memory block - ({largest.count} blocks, {largest.size/1024} Kb) ***")
	for l in largest.traceback.format():
    	    print(l)

	largest = snapshot.statistics("traceback")[2]
 
	print(f"\n*** Trace for largest memory block - ({largest.count} blocks, {largest.size/1024} Kb) ***")
	for l in largest.traceback.format():
    	    print(l)



from wrappers.EpisodeController import *
from wrappers.common_wrappers import *
from wrappers.reward_wrappers import *
from wrappers.loggers import *

from pretr_agent import make_agent

def make_iglu(*args, **kwargs):
    from gridworld.env import GridWorld
    from gridworld.tasks.task import Task
    custom_grid = np.ones((9, 11, 11))
    env = GridWorld(render=True, select_and_place=True, discretize=True, max_steps=900, fake=kwargs.get('fake', False))
    env.set_task(Task("", custom_grid, invariant=False))
    
    tg = RandomTargetGenerator(None, 0.01)
    sg = WalkingSubtaskGenerator()
    target = tg.get_target(None)
    sg.set_new_task(target)
    tc = TrainTaskController()
    sc = TrainSubtaskController()
    env = EpisodeController(env, tg, sg, tc, sc)
    #figure_generator = RandomFigure
    
    #env = TargetGenerator(env, fig_generator=figure_generator)
    #env = SubtaskGenerator(env)
    #env = VisualObservationWrapper(env)
    #env = JumpAfterPlace(env)

    #env = Discretization(env)
    #env = ColorWrapper(env)
    #env = RangetRewardFilledField(env)
    #env = Closeness(env)

    #env = SuccessRateFullFigure(env)
    #env = VideoLogger(env)
    #env = MultiAgentWrapper(env)
    #env = AutoResetWrapper(env)

    return env


tracemalloc.start(10)

env = make_iglu()

obs = env.reset()
i = 0
done = False
agent = make_agent()
n = 50000
while i < n:
    #a = agent.act((obs, ))
    obs, rew, done, info = env.step(env.action_space.sample())
    if done:
        obs = env.reset()
        print(i)
    i += 1
    if i % 1000 == 0:
        snapshot()

display_stats()
compare()
print_trace()
