import os
import numpy as np

import gym
from gridworld.tasks import DUMMY_TASK

from core.builder_walking import BuilderWalking

from gridworld.data import IGLUDataset

from nlp_model.agent import GridPredictor, get_dialog
from nlp_model.utils import plot_voxel, compute_metric

dataset = IGLUDataset(dataset_version='v0.1.0-rc1', )

FIXED_START_POSITION = np.array([0, 0, 0])

def eval_agent():
    grid_predictor = GridPredictor()
    env = gym.make('IGLUGridworld-v0', vector_state=True, render_size=(800, 600))
    # print(np.array(task.starting_grid).nonzero())
    env.set_task(DUMMY_TASK)
    obs = env.reset()
    total_score = []

    for j, (task_id, session_id, subtask_id, subtask) in enumerate(dataset):
        str_id = str(task_id) + '-session-' + str(session_id).zfill(3) + '-subtask-' + str(subtask_id).zfill(3)
        print('Starting task:', str_id)

        dialog = get_dialog(subtask)
        predicted_grid = grid_predictor.predict_grid(dialog)

        obs = env.reset()
        agent = BuilderWalking(predicted_grid, start_position=FIXED_START_POSITION)

        while not agent.is_done:
            action = agent.get_action()

            obs, reward, done, info = env.step(action)

        if not os.path.exists('plots'):
            os.makedirs('plots')

        f1_score = round(compute_metric(obs['grid'], subtask.target_grid)['completion_rate_f1'], 3)
        results = {'F1': f1_score}
        total_score.append(f1_score)
        results_str = " ".join([f"{metric}: {value}" for metric, value in results.items()])
        plot_voxel(obs['grid'], text=str_id + ' ' + f'({results_str})' + "\n" + dialog).savefig(
            f'./plots/{str_id}-built.png')
        plot_voxel(predicted_grid, text=str_id + ' ' + f'({results_str})' + "\n" + dialog).savefig(
            f'./plots/{str_id}-predicted.png')
        plot_voxel(subtask.target_grid, text=str_id + " (Ground truth)\n" + dialog).savefig(
            f'./plots/{str_id}-gt.png')

    print('Total F1 score:', np.mean(total_score))


if __name__ == "__main__":
    eval_agent()
