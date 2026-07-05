import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import nbformat
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "DRL_Battery_Energy_Storage.ipynb"
if not NOTEBOOK_PATH.exists():
    NOTEBOOK_PATH = REPO_ROOT / "DRL_BESS.ipynb"


class TestDRLBatteryNotebook(unittest.TestCase):
    @staticmethod
    def _load_notebook():
        return nbformat.read(NOTEBOOK_PATH, as_version=4)

    @staticmethod
    def _notebook_code():
        notebook = TestDRLBatteryNotebook._load_notebook()
        return "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "code"
        )

    @staticmethod
    def _exec_setup_through_environment():
        notebook = TestDRLBatteryNotebook._load_notebook()
        namespace = {}
        for cell in notebook.cells:
            source = "".join(cell.get("source", ""))
            if cell.get("cell_type") != "code":
                continue
            if "Transition = namedtuple" in source:
                break
            if source.startswith("# Quick environment check"):
                continue
            if source.startswith("# Step 3. Synthetic"):
                source = source.split("\nfig, ax1 =", 1)[0]
            with redirect_stdout(StringIO()):
                exec(source, namespace)
        return namespace

    @staticmethod
    def _exec_setup_through_rule_policy():
        notebook = TestDRLBatteryNotebook._load_notebook()
        namespace = TestDRLBatteryNotebook._exec_setup_through_environment()
        for cell in notebook.cells:
            source = "".join(cell.get("source", ""))
            if cell.get("cell_type") != "code":
                continue
            if source.startswith("def baseline_no_battery"):
                source = source.split("\nno_battery_cost =", 1)[0]
                with redirect_stdout(StringIO()):
                    exec(source, namespace)
                break
        return namespace

    def test_notebook_exists_with_expected_teaching_sections(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "DRL battery-storage notebook should exist")

        notebook = self._load_notebook()
        markdown = "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "markdown"
        )

        expected_sections = [
            "# Deep Reinforcement Learning for Battery Energy Storage Control",
            "## Step 0. Setup",
            "## Step 1. Problem Statement",
            "## Step 2. Reinforcement Learning Formulation",
            "## Step 3. Create Synthetic Power System Data",
            "## Step 4. Build the Battery Environment",
            "## Step 5. Define the DQN Agent",
            "## Step 6. Train the DQN",
            "## Step 7. Test the Learned Policy",
            "## Step 8. Visualize Results",
        ]

        for section in expected_sections:
            self.assertIn(section, markdown)

    def test_notebook_contains_core_dqn_building_blocks(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "DRL battery-storage notebook should exist")

        code = self._notebook_code()

        expected_snippets = [
            "class BatteryStorageEnv",
            "class DQN",
            "class ReplayMemory",
            "def select_action",
            "def train_one_step",
            "def rule_based_policy",
            "baseline_no_battery",
            "ACTIONS =",
            "epsilon",
        ]

        for snippet in expected_snippets:
            self.assertIn(snippet, code)

        self.assertNotIn("import gym", code)
        self.assertNotIn("from gym", code)

    def test_plot_uses_actual_initial_soc_instead_of_hard_coded_value(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "DRL battery-storage notebook should exist")

        code = self._notebook_code()

        self.assertIn("initial_soc_fraction=0.00", code)
        self.assertIn("initial_soc_fraction_used", code)
        self.assertNotIn("axhline(0.5", code)

    def test_visualization_compares_three_policies_across_key_quantities(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "DRL battery-storage notebook should exist")

        code = self._notebook_code()

        expected_snippets = [
            'no_battery_soc = np.zeros_like(HOURS, dtype=float)',
            'rule_soc = history_values(rule_result, "soc_fraction")',
            'dqn_soc = history_values(dqn_result, "soc_fraction")',
            'no_battery_power = np.zeros_like(HOURS, dtype=float)',
            'rule_power = history_values(rule_result, "battery_power_kw")',
            'dqn_power = history_values(dqn_result, "battery_power_kw")',
            'no_battery_grid = load_kw.astype(float)',
            'rule_grid = history_values(rule_result, "grid_power_kw")',
            'dqn_grid = history_values(dqn_result, "grid_power_kw")',
            'axes[1].step(HOURS, no_battery_soc',
            'axes[1].step(HOURS, rule_soc',
            'axes[1].step(HOURS, dqn_soc',
            'axes[2].step(HOURS, no_battery_power',
            'axes[2].step(HOURS, rule_power',
            'axes[2].step(HOURS, dqn_power',
            'axes[3].plot(HOURS, no_battery_grid',
            'axes[3].plot(HOURS, rule_grid',
            'axes[3].plot(HOURS, dqn_grid',
        ]

        for snippet in expected_snippets:
            self.assertIn(snippet, code)

    def test_output_labels_distinguish_energy_cost_from_reward_penalties(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "DRL battery-storage notebook should exist")

        notebook = self._load_notebook()
        code = self._notebook_code()
        markdown = "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "markdown"
        )

        self.assertIn("No battery grid-energy cost", code)
        self.assertIn("Straightforward rule grid-energy cost", code)
        self.assertIn("DQN grid-energy cost", code)
        self.assertIn("reported cost is grid-energy cost", markdown)
        self.assertIn("straightforward rule is intentionally naive", markdown)

    def test_text_explicitly_marks_load_and_price_as_synthetic(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "DRL battery-storage notebook should exist")

        notebook = self._load_notebook()
        markdown = "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "markdown"
        ).lower()

        self.assertIn("synthetic", markdown)
        self.assertIn("not measured field data", markdown)
        self.assertIn("not an official tariff", markdown)

    def test_battery_environment_uses_consistent_power_energy_units(self):
        namespace = self._exec_setup_through_environment()
        env = namespace["BatteryStorageEnv"](namespace["load_kw"], namespace["price_per_kwh"])
        load_kw = namespace["load_kw"]
        price_per_kwh = namespace["price_per_kwh"]

        state = env.reset()
        self.assertAlmostEqual(float(state[3]), 0.0, places=7)

        _, reward, done, info = env.step(2)
        expected_charge_kw = 15.0
        expected_soc_kwh = expected_charge_kw * env.efficiency * env.dt_hours
        expected_grid_power_kw = float(load_kw[0]) + expected_charge_kw
        expected_energy_cost = expected_grid_power_kw * float(price_per_kwh[0]) * env.dt_hours
        expected_reward = -(
            expected_energy_cost + env.cycling_cost * expected_charge_kw * env.dt_hours
        ) * env.reward_scale

        self.assertFalse(done)
        self.assertAlmostEqual(info["battery_power_kw"], expected_charge_kw, places=7)
        self.assertAlmostEqual(info["soc_kwh"], expected_soc_kwh, places=7)
        self.assertAlmostEqual(info["grid_power_kw"], expected_grid_power_kw, places=5)
        self.assertAlmostEqual(info["energy_cost"], expected_energy_cost, places=5)
        self.assertAlmostEqual(reward, expected_reward, places=5)

    def test_environment_keeps_physical_limits_over_action_sequence(self):
        namespace = self._exec_setup_through_environment()
        env = namespace["BatteryStorageEnv"](namespace["load_kw"], namespace["price_per_kwh"])
        load_kw = namespace["load_kw"]
        price_per_kwh = namespace["price_per_kwh"]

        self.assertEqual(len(load_kw), 24)
        self.assertEqual(len(price_per_kwh), 24)
        self.assertTrue(np.all(load_kw > 0.0))
        self.assertTrue(np.all(price_per_kwh > 0.0))

        env.reset()
        action_sequence = [2] * 8 + [0] * 12 + [1] * 4
        for action in action_sequence:
            _, _, done, info = env.step(action)
            step_info = env.history[-1]
            hour = step_info["hour"]

            self.assertGreaterEqual(step_info["soc_kwh"], -1e-9)
            self.assertLessEqual(step_info["soc_kwh"], env.capacity_kwh + 1e-9)
            self.assertGreaterEqual(step_info["soc_fraction"], -1e-9)
            self.assertLessEqual(step_info["soc_fraction"], 1.0 + 1e-9)
            self.assertLessEqual(abs(step_info["battery_power_kw"]), env.max_power_kw + 1e-9)
            self.assertGreaterEqual(step_info["grid_power_kw"], -1e-9)
            self.assertAlmostEqual(
                step_info["energy_cost"],
                step_info["grid_power_kw"] * float(price_per_kwh[hour]) * env.dt_hours,
                places=5,
            )
            if done:
                break

    def test_discharge_from_empty_battery_is_clipped_and_penalized(self):
        namespace = self._exec_setup_through_environment()
        env = namespace["BatteryStorageEnv"](namespace["load_kw"], namespace["price_per_kwh"])
        load_kw = namespace["load_kw"]
        price_per_kwh = namespace["price_per_kwh"]

        env.reset()
        _, reward, done, info = env.step(0)
        expected_energy_cost = float(load_kw[0]) * float(price_per_kwh[0]) * env.dt_hours
        expected_reward = -(
            expected_energy_cost + env.invalid_action_penalty
        ) * env.reward_scale

        self.assertFalse(done)
        self.assertAlmostEqual(info["battery_power_kw"], 0.0, places=7)
        self.assertAlmostEqual(info["soc_kwh"], 0.0, places=7)
        self.assertAlmostEqual(info["grid_power_kw"], float(load_kw[0]), places=5)
        self.assertAlmostEqual(reward, expected_reward, places=5)

    def test_rule_based_policy_cycles_between_full_and_empty_without_price(self):
        namespace = self._exec_setup_through_rule_policy()
        env = namespace["BatteryStorageEnv"](namespace["load_kw"], namespace["price_per_kwh"])
        rule_based_policy = namespace["rule_based_policy"]

        state = env.reset()
        actions = []
        soc_values = []
        for _ in range(12):
            action = rule_based_policy(state, env)
            actions.append(action)
            state, _, done, _ = env.step(action)
            soc_values.append(env.soc_kwh)
            self.assertFalse(done)

        self.assertEqual([2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2], actions)
        self.assertAlmostEqual(soc_values[4], env.capacity_kwh, places=7)
        self.assertAlmostEqual(soc_values[8], 0.0, places=7)

    def test_dqn_target_matches_one_step_bellman_update(self):
        code = self._notebook_code()
        self.assertIn("next_q_values = torch.zeros(BATCH_SIZE", code)
        self.assertIn("not_done = ~done_batch", code)
        self.assertIn("next_q_values[not_done] = target_net(next_state_batch[not_done]).max(dim=1).values", code)
        self.assertIn("target_q_values = reward_batch + GAMMA * next_q_values", code)

    def test_no_battery_baseline_matches_one_hour_energy_cost_formula(self):
        namespace = self._exec_setup_through_environment()
        load_kw = namespace["load_kw"]
        price_per_kwh = namespace["price_per_kwh"]
        expected_cost = float(np.sum(load_kw * price_per_kwh * 1.0))

        self.assertAlmostEqual(expected_cost, 191.5765, places=4)


if __name__ == "__main__":
    unittest.main()
