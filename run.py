"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import asyncio
import gym
from multiprocessing import Process, Queue

from tamer.agent import Tamer


#consumer
async def main():
    env = gym.make('MountainCar-v0')
    # hyperparameters
    discount_factor = 1
    epsilon = 0  # vanilla Q learning actually works well with no random exploration
    min_eps = 0
    num_episodes = 10
    tame = True  # set to false for vanilla Q learning

    # set a timestep for training TAMER
    # the more time per step, the easier for the human
    # but the longer it takes to train (in real time)
    # 0.2 seconds is fast but doable
    tamer_training_timestep = 0.3

    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame,
                  tamer_training_timestep, model_file_to_load=None)

    await agent.train(model_file_to_save='autosave')
    agent.play(n_episodes=1, render=True)
    agent.evaluate(n_episodes=30)


if __name__ == '__main__':
    asyncio.run(main())

    """TODO1: pickling error"""
    
    # q = Queue()

    # p1 = Process(target=asyncio.run(main()), args=())
    # # p2 = Process(target=h, args=(q,))
    
    # # p2.start()
    # p1.start()


    # p1.join()
    # # p2.join()



#Tried a solution from this stack overflow, but didn't work well: https://stackoverflow.com/questions/50550905/discord-python-bot-attributeerror-cant-pickle-local-object


# class Process(multiprocessing.Process):

#     def __init__(self,
#                  group=None,
#                  target=None,
#                  name=None,
#                  args=(),
#                  kwargs=None):
#         super().__init__(group, target, name, args, kwargs)
#         self.loop: typing.Optional[asyncio.AbstractEventLoop] = None
#         self.stopped: typing.Optional[asyncio.Event] = None

#     def run(self):
#         self.loop = asyncio.get_event_loop()
#         self.stopped = asyncio.Event()
#         self.loop.run_until_complete(self._run())
    
#     async def _run(self):
#         print('hi')
#         # """My async stuff here"""
#         # #consumer
#         # env = gym.make('MountainCar-v0')
#         # # hyperparameters
#         # discount_factor = 1
#         # epsilon = 0  # vanilla Q learning actually works well with no random exploration
#         # min_eps = 0
#         # num_episodes = 10
#         # tame = True  # set to false for vanilla Q learning

#         # # set a timestep for training TAMER
#         # # the more time per step, the easier for the human
#         # # but the longer it takes to train (in real time)
#         # # 0.2 seconds is fast but doable
#         # tamer_training_timestep = 0.3

#         # agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame,
#         #               tamer_training_timestep, model_file_to_load=None)

#         # await agent.train(model_file_to_save='autosave')
#         # agent.play(n_episodes=1, render=True)
#         # agent.evaluate(n_episodes=30)

# if __name__ == '__main__':
#     myProcess = Process()
#     asyncio.run(myProcess.run())



