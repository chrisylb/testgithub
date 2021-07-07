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
#from highway_env.vehicle.other_vehicle import OtherVehicle
import pprint

class MergeEgoEnv1(AbstractEnv):
    COLLISION_REWARD: float = -10
    stay_on_lane_REWARD: float = 1
    get_goal=False
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
                "steering_range": [-np.pi / 3, np.pi / 3],
            },
               "duration": 20,
               'simulation_frequency': 10,
               'policy_frequency': 1,
               'other_vehicles_type': 'highway_env.vehicle.concontroller.ConcontrolledVehicle'

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

        if self.vehicle.stay_on_lane():
            stay_on_lane_reward=0.01
        else:
            stay_on_lane_reward=-0.02
        if self.vehicle.position[0]>155 and self.vehicle.stay_on_lane():

            goal=10
            self.get_goal=True
        else:
            goal=0

        reward = self.COLLISION_REWARD * self.vehicle.crashed+self.stay_on_lane_REWARD*stay_on_lane_reward+goal
        return reward
        #return utils.lmap( reward,[-10,1],[0, 1])


    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.get_goal or self.vehicle.position[1]<15 or self.vehicle.position[1]>28

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lke = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lke)
        #net.add_lane("e", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        #road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

        Make a road composed of a straight highway and a merging lane.

        :return: the road
 """
        net = RoadNetwork()
        # Highway lanes
        ends = [90,155,180]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [18.9,21.2,25.5]
        #line_type = [[c, s], [n, c]]
        #line_type_merge=[[s,s],[n,c]]
        line_type_merge=[[c,n],[s,c]]
        #line_type_highway=[[c,s],[n,c]]
        #line_type_merge = [[c, s], [n, s]]
        line_type = [c, c]
        #amplitude = 2
        #ljk = StraightLane([0,6.85],[183,6.85],line_types=[c,c], forbidden=True)
        #lbc = SineLane(ljk.position(182,amplitude), ljk.position(218, amplitude),
        #               -amplitude, 2 * np.pi / 72, np.pi / 2, line_types=[c, n], forbidden=True)
        #net.add_lane("b", "c", lbc)
        #net.add_lane("a", "b", StraightLane([0, 6.85], [183, 6.85], line_types=[c,n]))
        '''
        for i in range(2):
            net.add_lane("a", "c", StraightLane([0, y[i]], [218, y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([218, y[i]], [400, y[i]], line_types=line_type_highway[i]))
        #net.add_lane("e", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        '''
        amplitude=2.15

        lmn=StraightLane([0,y[0]],[90, y[0]],2.2,line_types=[c,n], forbidden=True)
        lab=StraightLane([0,y[1]],[90, y[1]],2.2,line_types=[c,n], forbidden=True)
        net.add_lane("m", "n", StraightLane([0, y[0]], [90, y[0]],2.2, line_types=line_type_merge[0]))
        net.add_lane("a", "b", StraightLane([0, y[1]], [90, y[1]],2.2, line_types=line_type_merge[1]))
        net.add_lane("d", "e", StraightLane([155, y[2]], [180, y[2]],2.2, line_types=line_type))
        lnd = SineLane(lmn.position(90,amplitude), lmn.position(155, amplitude),
                       -amplitude, 2 * np.pi / 130, np.pi / 2,2.2, line_types=[c,n], forbidden=True)
        lbd = SineLane(lab.position(90,amplitude), lab.position(155, amplitude),
                       -amplitude, 2 * np.pi / 130, np.pi / 2,2.2, line_types=[s,c], forbidden=False)
        net.add_lane("n", "d", lnd)
        net.add_lane("b","d",lbd)
        #road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road
        road.objects.append(Obstacle(road, lnd.position(65, 0)))
    def _make_vehicles(self) -> None:

        road = self.road
        #add random ego vechicle
        random_es = 4.5 * np.random.randint(-1,1)+6
        ego_vehicle = self.action_type.vehicle_class(road,
                                                    road.network.get_lane(("m", "n", 0)).position(60, 0), speed=random_es)
        road.vehicles.append(ego_vehicle)
        #try:
        #    ego_vehicle.plan_route_to("d")
        #except AttributeError:
        #    pass
        self.vehicle = ego_vehicle
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        spead=np.random.randint(-1,1)
        other_vehicle1=other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(70, 0), speed=6 + (4 * spead))
        other_vehicle2=other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(55, 0), speed=5 + (3 * spead))
        other_vehicle3 = other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(35, 0),
                                             speed=5 + (3 * spead))
        #other_vehicle3 = other_vehicles_type(road, road.network.get_lane(("b", "d", 0)).position(0, 0),speed=4 + (3 * spead))

        road.vehicles.append(other_vehicle1)
        road.vehicles.append(other_vehicle2)
        road.vehicles.append(other_vehicle3)
        other_vehicle1.plan_route_to("e")
        other_vehicle2.plan_route_to("e")
        other_vehicle3.plan_route_to("e")
        #road.vehicles.append(other_vehicle3)

register(
    id='merge-v3',
    entry_point='highway_env.envs:MergeEgoEnv1',
)
