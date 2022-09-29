import copy
import enum
from collections import deque, defaultdict
from typing import Optional, Deque, Tuple, Dict
import numpy as np


class Action(enum.IntEnum):
    PASS = 0
    STEP_FORWARD = 1
    STEP_BACKWARD = 2
    STEP_LEFT = 3
    STEP_RIGHT = 4
    JUMP = 5
    INVENTORY_BLUE = 6
    INVENTORY_GREEN = 7
    INVENTORY_RED = 8
    INVENTORY_ORANGE = 9
    INVENTORY_PURPLE = 10
    INVENTORY_YELLOW = 11
    CAMERA_LEFT = 12
    CAMERA_RIGHT = 13
    CAMERA_UP = 15
    CAMERA_DOWN = 14
    BREAK_BLOCK = 16
    PLACE_BLOCK = 17


from core.colors import Colors


class Direction(enum.IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


CAMERA_ACTION_BY_DIRECTION = {
    Direction.UP: Action.CAMERA_UP,
    Direction.DOWN: Action.CAMERA_DOWN,
    Direction.LEFT: Action.CAMERA_LEFT,
    Direction.RIGHT: Action.CAMERA_RIGHT
}

DIRECTION_TO_MOVEMENT_ACTION = {
    Direction.UP: Action.STEP_FORWARD,
    Direction.DOWN: Action.STEP_BACKWARD,
    Direction.LEFT: Action.STEP_LEFT,
    Direction.RIGHT: Action.STEP_RIGHT
}

COLOR_TO_INVENTORY_ACTION = {
    Colors.BLUE: Action.INVENTORY_BLUE,
    Colors.GREEN: Action.INVENTORY_GREEN,
    Colors.RED: Action.INVENTORY_RED,
    Colors.ORANGE: Action.INVENTORY_ORANGE,
    Colors.PURPLE: Action.INVENTORY_PURPLE,
    Colors.YELLOW: Action.INVENTORY_YELLOW
}


class BuilderWalking:
    def __init__(self,
                 target_grid: np.ndarray,
                 current_position: np.ndarray,
                 block_steps=4,
                 camera_rotation_degrees=5):
        self.initialized = False
        self.target_grid = self._reshape_grid(self.swap_grid_axes(target_grid))
        self.camera = [0, 0]
        self.block_steps = block_steps
        self.camera_rotation_degrees = camera_rotation_degrees
        self.current_position, self.y_position = self.process_current_position(current_position)
        self.queue: Deque[Action] = deque()
        self.building_queue: Deque[BuildingQueueItem] = deque()
        self.is_done = False

    def build_under(self, color: Optional[Colors] = None):
        if color is None:
            color = Colors.BLUE
        self.add_action(COLOR_TO_INVENTORY_ACTION[color])
        self.camera_down()
        self.skip_action()
        self.add_action(Action.JUMP)
        self.repeat_action(Action.PLACE_BLOCK, 5)
        self.skip_action(5)
        self.y_position += 1

    def destroy_under(self):
        self.camera_down()
        self.add_action(Action.BREAK_BLOCK)
        self.skip_action(6)
        self.y_position -= 1

    def swap_grid_axes(self, grid: np.ndarray):
        new_grid = copy.deepcopy(grid)
        new_grid = new_grid.swapaxes(0, 2)
        return new_grid

    def process_current_position(self, current_position: np.ndarray):
        if current_position is None:
            return None, 0
        plane_position = np.zeros(2, dtype=np.int32)
        plane_position[0], plane_position[1], y_position = current_position[2], \
                                                           current_position[0], \
                                                           current_position[1]
        return self._add_bias_to_position(plane_position + np.array([5, 5])), int(y_position)

    def _reshape_grid(self, target_grid):
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

    def _add_bias_to_position(self, target: np.ndarray):
        return target + np.array([2, 2])

    def add_action(self, action: Action):
        self.queue.append(action)

    def repeat_action(self, action: Action, amount: int):
        for _ in range(amount):
            self.add_action(action)

    def _move_steps(self, steps: int, direction: Direction):
        steps = max(steps, 0)
        for _ in range(steps):
            self.add_action(DIRECTION_TO_MOVEMENT_ACTION[direction])

    def _rotate(self, steps: int, direction: Direction):
        for _ in range(steps):
            # if CAMERA_ACTION_BY_DIRECTION[direction]
            self.add_action(CAMERA_ACTION_BY_DIRECTION[direction])

    def _resolve_direction(self, sign_direction_vector):
        directions = np.zeros(2)
        directions[0] = Direction.UP if sign_direction_vector[0] > 0 else Direction.DOWN
        directions[1] = Direction.RIGHT if sign_direction_vector[1] > 0 else Direction.LEFT
        return directions

    def move_to_position(self, target: np.ndarray, add_bias=True):
        if add_bias:
            target = self._add_bias_to_position(target)
        if np.array_equal(target, self.current_position):
            return
        self.camera_forward()
        steps = (target - self.current_position) * self.block_steps
        directions = self._resolve_direction(np.sign(steps) * np.array([-1, 1]))
        steps = np.abs(steps)
        self._move_steps(steps[1], directions[1])
        self._move_steps(steps[0], directions[0])
        self.current_position = target

    def move_camera(self, target: np.ndarray):
        if np.array_equal(target, self.current_position):
            return
        modulus = target % self.camera_rotation_degrees
        assert np.alltrue(modulus == 0), (f"Can't rotate to this position, because "
                                          f"environment rotate camera by "
                                          f"{self.camera_rotation_degrees} degrees on each step")

        steps = (target - self.camera) / self.camera_rotation_degrees
        directions_sign = np.sign(steps)
        directions = self._resolve_direction(directions_sign[::-1])[::-1]
        steps = np.abs(steps)
        self._rotate(int(steps[0]), directions[0])
        self._rotate(int(steps[1]), directions[1])
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
            self.initialized = True
            self.build_building_queue()
        if not self.queue:
            if self.building_queue:
                self.building_queue.popleft().process()
            else:
                self.is_done = True
                return Action.PASS
        return self.queue.popleft()

    def _handle_column(self, position: Tuple[int, int]):
        height_to_build_block_color = defaultdict()
        max_position = -1
        i, k = position
        for j in range(self.target_grid.shape[2]):
            if self.target_grid[i, k, j] > 0:
                max_position = j
                height_to_build_block_color[j] = self.target_grid[i, k, j]

        if max_position > -1:
            self.add_to_building_queue((i, k), max_position + 1, height_to_build_block_color)

    def add_to_building_queue(self,
                              position: Tuple[int, int],
                              height: int,
                              height_to_build_block_color: Dict[int, Colors]):

        self.building_queue.append(BuildingQueueItem(position, height, height_to_build_block_color, self))

    def skip_action(self, amount: int = 1):
        for _ in range(amount):
            self.add_action(Action.PASS)

    def build_tower(self, height: int, height_to_build_block_color: Dict[int, Colors]):
        for k in range(height):
            color = None
            if k in height_to_build_block_color:
                color = height_to_build_block_color[k]
            self.build_under(color)

    def _resolve_skip_when_falling(self, k):
        if k < 2 or k == 3:
            self.skip_action(2)
        else:
            self.skip_action()

    def _resolve_break_when_falling(self, k):
        if k < 2 or k == 3:
            self.add_action(Action.BREAK_BLOCK)
            self.skip_action()
        else:
            self.add_action(Action.BREAK_BLOCK)

    def fall_from_tower(self, height_to_build_block_color: Dict[int, Colors]):
        if self.y_position > 0:
            self.camera_forward()
            self.skip_action(2)
            self._move_steps(4, Direction.DOWN)
            self.skip_action(7)
            for k in range(self.y_position - 2, 1, -1):
                if k in height_to_build_block_color:
                    self._resolve_skip_when_falling(self.y_position - 2 - k)
                else:
                    self._resolve_break_when_falling(self.y_position - 2 - k)
            if 1 not in height_to_build_block_color:
                self.skip_action()
                self.add_action(Action.BREAK_BLOCK)
            if 0 not in height_to_build_block_color:
                self.move_camera(np.array([self.camera[0], -65]))
                self.add_action(Action.BREAK_BLOCK)
            self.camera_forward()
            self.y_position = 0
            self.current_position[0] += 1

    def build_building_queue(self):
        for i in range(2, self.target_grid.shape[0] - 2):
            for k in range(2, self.target_grid.shape[1] - 2):
                self._handle_column((i, k))


class BuildingQueueItem:
    __slots__ = 'position', 'height', 'height_to_build_block_color', 'agent'

    def __init__(self,
                 position: Tuple[int, int],
                 height: int,
                 height_to_build_block_color: Dict[int, Colors],
                 agent: BuilderWalking):
        self.position = position
        self.height = height
        self.height_to_build_block_color = height_to_build_block_color
        self.agent = agent

    def process(self):
        self.agent.move_to_position(np.asarray(self.position), add_bias=False)
        self.agent.build_tower(self.height, self.height_to_build_block_color)
        self.agent.fall_from_tower(self.height_to_build_block_color)
