import numpy as np
from numpy.core._multiarray_umath import ndarray
#from gym import GoalEnv
from gym.envs.registration import register
from typing import Tuple
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle
#from highway_env.vehicle.concontroller import ConcontrolledVehicle
from highway_env.vehicle.other_vehicle import OtherVehicle
import pprint

class MergeEgoEnv(AbstractEnv):
    COLLISION_REWARD: float = -10
    HIGH_SPEED_REWARD: float = 1
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 4,
                "absolute": True,
                "normalize":False,
                "features":['x', 'y', 'vx', 'vy']
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "acceleration_range": [-1,1],
                "steering_range": [-np.pi / 6, np.pi / 6],
            },
               "duration": 20,
               'simulation_frequency': 25,
               'policy_frequency': 1,

        })
       # 'other_vehicles_type': 'highway_env.vehicle.other_vehicle.OtherVehicle'
         #'other_vehicles_type': 'highway_env.vehicle.kinematics.Vehicle'
        #'other_vehicles_type': 'highway_env.vehicle.kinematics.Vehicle'
        return config
    def _reward(self, action: np.ndarray):
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        #change velocity reward
        if self.vehicle.speed< 33.3 and self.vehicle.speed > 13:
            high_speed_reward=0.01
        else:
            high_speed_reward=-0.02
        if self.vehicle.position[0]>218:

            safe=5
        else:
            safe=0

        reward = self.COLLISION_REWARD * self.vehicle.crashed+self.HIGH_SPEED_REWARD*high_speed_reward+safe
        return reward
        #return utils.lmap( reward,[-10,1],[0, 1])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        if self.vehicle.position[0]>218:
            done=True
        return obs, reward, done, info

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.vehicle.position[0] > 218

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road

        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 20, 80]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lke = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lke)
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        #net.add_lane("e", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        #road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road



        Make a road composed of a straight highway and a merging lane.

        :return: the road
 """
        net = RoadNetwork()

        # Highway lanes
        ends = [400, 217, 182, 0]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [10.85,14.85]
        #line_type = [[c, s], [n, c]]
        line_type_merge=[[s,s],[n,c]]
        line_type_highway=[[c,s],[n,c]]
        #line_type_merge = [[c, s], [n, s]]
        amplitude = 2
        ljk = StraightLane([0,6.85],[183,6.85],line_types=[c,c], forbidden=True)
        lbc = SineLane(ljk.position(182,amplitude), ljk.position(218, amplitude),
                       -amplitude, 2 * np.pi / 72, np.pi / 2, line_types=[c, n], forbidden=True)
        net.add_lane("b", "c", lbc)
        net.add_lane("a", "b", StraightLane([0, 6.85], [183, 6.85], line_types=[c,n]))
        for i in range(2):
            net.add_lane("a", "c", StraightLane([0, y[i]], [218, y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([218, y[i]], [400, y[i]], line_types=line_type_highway[i]))

        #net.add_lane("e", "c", lbc)
        #net.add_lane("a","b",StraightLane([400,4.85],[218,4.85],line_types=line_type_merge,forbidden=True))
        #net.add_lane("a","c",StraightLane([400,8.85],[218,8.85],line_types=line_type_highway))
        #net.add_lane("b","c",StraightLane([218,4.85],[182,4.85],line_type=line_type_merge))
        #net.add_lane("b","c",StraightLane([218,8.85],[182,8.85],line_type=line_type_highway))
        #net.add_lane("c","d",StraightLane([182,8.85],[0,8.85],line_types=line_type_merge))
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        #road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:

        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                    road.network.get_lane(("a", "b", 0)).position(10, 0), speed=13)
        road.vehicles.append(ego_vehicle)
        #try:
        #    ego_vehicle.plan_route_to("d")
        #except AttributeError:
        #    pass
        self.vehicle = ego_vehicle
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        spead=self.np_random.randn()
       # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "c", 0)).position(200, 0), speed=26+(4*spead)))
       # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "c", 0)).position(140, 0), speed=26+(3*spead)))

register(
    id='merge-v2',
    entry_point='highway_env.envs:MergeEgoEnv',
)
