import copy
import enum
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Union
import numpy as np

from core.colors import Colors

FLYING_DISTANCE_Y = 0.34641016
FLYING_DISTANCES = (0.75, 0.53033006)
INERTIA = 10



class Directions(enum.IntEnum):
    ABOVE = 1
    BELOW = 2
    NORTH = 3
    EAST = 4
    WEST = 5
    SOUTH = 6


@dataclass
class BuildBlockContext:
    position: np.ndarray
    color: Colors


@dataclass
class DeleteBlockContext:
    position: np.ndarray


BuildContext = Union[DeleteBlockContext, BuildBlockContext]

"""
!!! WARNING NOT PROPERLY WORKING !!!
"""

class BuilderFlying:
    def __init__(self,
                 target_grid: np.ndarray,
                 current_grid: Optional[np.ndarray] = None,
                 current_position: Optional[np.ndarray] = None):
        self.initialized = False

        self.target_grid = self.reshape_grid(self.swap_grid_axes(target_grid))
        self.current_grid = np.zeros(self.target_grid.shape) if current_grid is None \
            else self.reshape_grid(self.swap_grid_axes(current_grid))
        self.camera = np.zeros(2)
        self.current_position = current_position if current_position is None \
            else self.process_current_position(current_position)
        self.queue = deque()
        self.building_queue: Deque[BuildQueueItem] = deque()
        self.is_done = False

    def swap_grid_axes(self, grid: np.ndarray):
        new_grid = copy.deepcopy(grid)
        new_grid = new_grid.swapaxes(0, 2)
        return new_grid

    def process_current_position(self, current_position: np.ndarray):
        current_position[0], current_position[1], current_position[2] = current_position[2], \
                                                                        current_position[0], \
                                                                        current_position[1]
        return self.add_bias_to_position(current_position + np.array([5, 5, 0]))
    def reshape_grid(self, target_grid):
        target_grid_enchanted = np.concatenate([np.full((2, *target_grid.shape[1:]), -1),
                                                target_grid,
                                                np.full((2, *target_grid.shape[1:]), -1)], axis=0)
        target_grid_enchanted = np.concatenate([np.full((target_grid_enchanted.shape[0],
                                                         2,
                                                         target_grid_enchanted.shape[2]), -1),
                                                target_grid_enchanted,
                                                np.full((target_grid_enchanted.shape[0],
                                                         2,
                                                         target_grid_enchanted.shape[2]), -1)], axis=1)

        return target_grid_enchanted

    def place_block(self, color: Colors):
        action = self.get_default_action()
        action['inventory'] = color
        action['placement'] = 1
        self.add_action(action)

    def delete_block(self):
        action = self.get_default_action()
        action['inventory'] = 0
        action['placement'] = 2
        self.add_action(action)

    def get_flying_distances(self, movement: np.ndarray):
        flying_distances = np.array([FLYING_DISTANCES[0], FLYING_DISTANCES[0], FLYING_DISTANCE_Y])
        number_of_moving_axis = np.array(movement[:2].nonzero()).size - 1
        if number_of_moving_axis == 1:
            flying_distances[:2] = np.full(2, FLYING_DISTANCES[number_of_moving_axis])

        return flying_distances

    def get_diff_for_x_z_axes(self, abs_diff: np.ndarray):
        min_axis = np.argmin(abs_diff)
        max_axis = 1 if min_axis == 0 else 0
        if abs_diff[min_axis] == 0 or abs_diff[max_axis] == 0:
            return abs_diff, None, None
        diff_left = abs_diff[max_axis] - abs_diff[min_axis]
        new_diff = copy.deepcopy(abs_diff)
        new_diff[max_axis] -= diff_left
        return new_diff, diff_left, max_axis

    def add_action(self, action):
        self.queue.append(action)
        for _ in range(INERTIA):
            self.queue.append(self.get_default_action())

    def get_step_enchantment(self, diff, steps, flying_distance, k = 0.95):
        distance = steps * flying_distance - diff
        if distance >= flying_distance * k:
            return -1
        elif distance <= -flying_distance * k:
            return 1
        else:
            return 0

    def resolve_number_of_steps(self, abs_diff):
        flying_distances = self.get_flying_distances(abs_diff)
        steps_vec = np.ceil(abs_diff / flying_distances)
        enchanted_steps = np.array([steps + self.get_step_enchantment(abs_diff[i], steps, flying_distances[i])
                                    for i, steps in enumerate(steps_vec)])
        return enchanted_steps

    def add_bias_to_position(self, target: np.ndarray):
        return target + np.array([2, 2, 0])

    # TODO, if no initial position provided
    def center_position(self):
        self.current_position = np.array([11, 11, 9])
        self.move_to_position(np.array([0, 0, 0]))
        self.current_position = np.array([2, 2, 0])

    def move_to_position(self,
                         target: np.ndarray,
                         return_camera_position=False,
                         add_bias=True):
        if add_bias:
            target = self.add_bias_to_position(target)
        if np.array_equal(target, self.current_position):
            return
        last_camera_position = self.camera
        self.camera_forward()
        self.camera_to_north()

        diff = target - self.current_position
        print(target)
        abs_diff = np.abs(diff)
        direction = np.sign(diff)
        abs_diff[:2], diff_left, axis_left = self.get_diff_for_x_z_axes(abs_diff[:2])
        steps = self.resolve_number_of_steps(abs_diff)
        max_step_axis = np.argmax(steps)
        other_axis_1, other_axis_2 = 1, 2
        if max_step_axis == 1:
            other_axis_1, other_axis_2 = 0, 2
        elif max_step_axis == 2:
            other_axis_1, other_axis_2 = 0, 1
        while steps[max_step_axis] > 0:
            action = self.get_default_action()
            if steps[other_axis_1] > 0:
                action['movement'][other_axis_1] = direction[other_axis_1]
                steps[other_axis_1] -= 1
            if steps[other_axis_2] > 0:
                action['movement'][other_axis_2] = direction[other_axis_2]
                steps[other_axis_2] -= 1
            action['movement'][max_step_axis] = direction[max_step_axis]
            steps[max_step_axis] -= 1
            self.add_action(action)
        # TODO make more optimal, so it runs with y movement
        if diff_left and diff_left > 0:
            steps = int(diff_left // FLYING_DISTANCES[0])
            steps += self.get_step_enchantment(diff_left, steps, FLYING_DISTANCES[0])
            for _ in range(steps):
                action = self.get_default_action()
                action['movement'][axis_left] = direction[axis_left]
                self.add_action(action)
        self.current_position = target
        if return_camera_position:
            self.move_camera(last_camera_position)

    def get_default_action(self):
        return {
            'movement': np.zeros(3, dtype=np.float32),
            'camera': np.zeros(2, dtype=np.float32),
            'inventory': 0,
            'placement': 0
        }

    def move_camera(self, target: np.ndarray):
        action = self.get_default_action()
        action['camera'] = target - self.camera
        self.add_action(action)
        self.camera = target

    def camera_to_north(self):
        if self.camera[0] == 0:
            return
        self.move_camera(np.array([0, self.camera[1]]))

    def camera_to_south(self):
        if self.camera[0] == 180:
            return
        self.move_camera(np.array([180, self.camera[1]]))

    def camera_to_east(self):
        if self.camera[0] == 90:
            return
        self.move_camera(np.array([90, self.camera[1]]))

    def camera_to_west(self):
        if self.camera[0] == -90:
            return
        self.move_camera(np.array([-90, self.camera[1]]))

    def camera_down(self):
        if self.camera[1] == -90:
            return
        self.move_camera(np.array([self.camera[0], -90]))

    def camera_forward(self):
        if self.camera[1] == 0:
            return
        self.move_camera(np.array([self.camera[0], 0]))

    def camera_up(self):
        if self.camera[1] == 90:
            return
        self.move_camera(np.array([self.camera[0], 90]))

    def get_action(self):
        if not self.initialized:
            self.build_building_queue()
            self.initialized = True
        if not self.queue:
            if self.building_queue:
                self.building_queue.popleft().process()
            else:
                self.is_done = True
                return self.get_default_action()
        return self.queue.popleft()

    def build_building_queue(self):
        for i in range(2, self.target_grid.shape[0] - 2):
            for j in range(self.target_grid.shape[2]):
                for k in range(2, self.target_grid.shape[1] - 2):
                    if self.target_grid[i, k, j] == -1:
                        continue
                    if self.target_grid[i, k, j] != self.current_grid[i, k, j]:
                        if self.current_grid[i, k, j] == 0:
                            self.build_block_basic(np.array([i, k, j]), self.target_grid[i, k, j])

    def build_block_basic(self, position, color):
        x, z, y = position
        if y > 0 and self.current_grid[x, z, y - 1] == 0:
            build_queue = []
            remove_queue = []
            for j in range(y):
                current_position = (x, z, j)
                if self.current_grid[current_position] != 0:
                    build_queue = []
                    remove_queue = []
                    continue

                build_context = BuildBlockContext(np.asarray(current_position), color)
                build_queue.append(BuildQueueItem(self, build_context))
                remove_context = DeleteBlockContext(np.asarray(position))
                remove_queue.append(BuildQueueItem(self, remove_context))
            self.building_queue.extend(build_queue)
            build_context = BuildBlockContext(np.asarray(position), color)
            self.building_queue.append(BuildQueueItem(self, build_context))
            self.building_queue.extend(remove_queue)
            self.current_grid[tuple(position)] = color
            return

        build_context = BuildBlockContext(np.asarray(position), color)
        self.building_queue.append(BuildQueueItem(self, build_context))
        self.current_grid[tuple(position)] = color


class BuildQueueItem:
    __slots__ = 'agent', 'context'

    def __init__(self,
                 agent: BuilderFlying,
                 context: BuildContext):
        self.agent = agent
        self.context = context

    def build_block(self):
        position = copy.deepcopy(self.context.position)
        position[2] += 1
        self.agent.move_to_position(position, add_bias=False)
        self.agent.camera_down()
        self.agent.place_block(self.context.color)

    def delete_block(self):
        position = copy.deepcopy(self.context.position)
        position[0] += 1
        self.agent.move_to_position(position, add_bias=False)
        self.agent.camera_to_north()
        self.agent.delete_block()
        self.agent.current_grid[tuple(self.context.position)] = 0

    def process(self):
        if isinstance(self.context, BuildBlockContext):
            self.build_block()
        elif isinstance(self.context, DeleteBlockContext):
            self.delete_block()
