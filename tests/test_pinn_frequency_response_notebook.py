import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import nbformat
import numpy as np


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "PINN_Frequency.ipynb"


class TestPINNSwingEquationNotebook(unittest.TestCase):
    @staticmethod
    def _load_notebook():
        return nbformat.read(NOTEBOOK_PATH, as_version=4)

    @staticmethod
    def _notebook_code():
        notebook = TestPINNSwingEquationNotebook._load_notebook()
        return "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "code"
        )

    @staticmethod
    def _notebook_markdown():
        notebook = TestPINNSwingEquationNotebook._load_notebook()
        return "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "markdown"
        )

    @staticmethod
    def _markdown_section(start_heading, end_heading):
        markdown = TestPINNSwingEquationNotebook._notebook_markdown()
        start = markdown.index(start_heading)
        end = markdown.index(end_heading, start)
        return markdown[start:end]

    @staticmethod
    def _exec_setup_through_reference_solution():
        notebook = TestPINNSwingEquationNotebook._load_notebook()
        namespace = {}
        for cell in notebook.cells:
            source = "".join(cell.get("source", ""))
            if cell.get("cell_type") != "code":
                continue
            if source.startswith("# Step 4. Sparse"):
                break
            if "plt.show()" in source:
                source = source.split("\nfig", 1)[0]
            with redirect_stdout(StringIO()):
                exec(source, namespace)
        return namespace

    @staticmethod
    def _exec_setup_through_sparse_data():
        notebook = TestPINNSwingEquationNotebook._load_notebook()
        namespace = {}
        for cell in notebook.cells:
            source = "".join(cell.get("source", ""))
            if cell.get("cell_type") != "code":
                continue
            if source.startswith("# Step 5. Ordinary"):
                break
            if source.startswith("class MLP"):
                break
            if "plt.show()" in source:
                source = source.split("\nfig", 1)[0]
            with redirect_stdout(StringIO()):
                exec(source, namespace)
        return namespace

    def test_notebook_exists_with_expected_teaching_sections(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")

        notebook = self._load_notebook()
        markdown = "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "markdown"
        )

        expected_sections = [
            "# Physics-Informed Neural Network for the Swing Equation",
            "## Step 0. Setup",
            "## Step 1. Problem Statement",
            "## Step 2. Governing Physics",
            "## Step 3. Generate Reference Solution",
            "## Step 4. Create Sparse Measurement Data",
            "## Step 5. Ordinary Neural Network Baseline",
            "## Step 6. Define the PINN Model",
            "## Step 7. Define PINN Loss",
            "## Step 8. Train the PINN",
            "## Step 9. Evaluate and Compare",
            "## Step 10. Discussion for Power Systems",
            "## References",
        ]

        for section in expected_sections:
            self.assertIn(section, markdown)

    def test_notebook_contains_core_swing_pinn_building_blocks(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")

        code = self._notebook_code()
        expected_snippets = [
            "def swing_response_exact",
            "class MLP",
            "def physics_residual",
            "torch.autograd.grad",
            "delta_hat",
            "omega_hat",
            "r_delta",
            "r_omega",
            "data_loss",
            "physics_loss",
            "ic_loss",
            "total_loss",
            "lambda_physics",
            "lambda_ic",
            "Ks",
        ]

        for snippet in expected_snippets:
            self.assertIn(snippet, code)

    def test_analytical_solution_satisfies_linearized_swing_equation(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")
        namespace = self._exec_setup_through_reference_solution()
        swing_response_exact = namespace["swing_response_exact"]
        M = namespace["M"]
        D = namespace["D"]
        Ks = namespace["Ks"]
        delta_p = namespace["delta_p"]

        t = np.linspace(0.0, 20.0, 400)
        delta, omega = swing_response_exact(t, M, D, Ks, delta_p)
        d_delta_dt = omega

        omega_n = np.sqrt(Ks / M)
        zeta = D / (2.0 * np.sqrt(M * Ks))
        omega_d = omega_n * np.sqrt(1.0 - zeta**2)
        alpha = zeta * omega_n
        delta_ss = delta_p / Ks
        d_omega_dt = (
            delta_ss
            * (omega_n**2 / omega_d)
            * np.exp(-alpha * t)
            * (omega_d * np.cos(omega_d * t) - alpha * np.sin(omega_d * t))
        )

        r_delta = d_delta_dt - omega
        r_omega = M * d_omega_dt + D * omega + Ks * delta - delta_p

        self.assertAlmostEqual(float(delta[0]), 0.0, places=12)
        self.assertAlmostEqual(float(omega[0]), 0.0, places=12)
        self.assertLess(float(np.max(np.abs(r_delta))), 1e-12)
        self.assertLess(float(np.max(np.abs(r_omega))), 1e-12)

    def test_parameters_are_underdamped_and_response_oscillates(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")
        namespace = self._exec_setup_through_reference_solution()
        swing_response_exact = namespace["swing_response_exact"]
        M = namespace["M"]
        D = namespace["D"]
        Ks = namespace["Ks"]
        delta_p = namespace["delta_p"]

        self.assertGreater(M, 0.0)
        self.assertGreater(D, 0.0)
        self.assertGreater(Ks, 0.0)
        self.assertGreater(delta_p, 0.0)

        damping_ratio = D / (2.0 * np.sqrt(M * Ks))
        self.assertGreater(damping_ratio, 0.0)
        self.assertGreaterEqual(damping_ratio, 0.20)
        self.assertLess(damping_ratio, 0.50)
        self.assertLess(damping_ratio, 1.0)

        t = np.linspace(0.0, 20.0, 400)
        delta, omega = swing_response_exact(t, M, D, Ks, delta_p)
        steady_delta = delta_p / Ks
        centered_delta = delta - steady_delta

        self.assertGreater(int(np.sum(np.diff(np.signbit(centered_delta)) != 0)), 1)
        self.assertGreater(int(np.sum(np.diff(np.signbit(omega)) != 0)), 1)
        self.assertLess(abs(float(delta[-1] - steady_delta)), abs(float(delta[100] - steady_delta)))

    def test_step_2_directly_presents_state_equations_and_symbol_definitions(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")
        step_2 = self._markdown_section("## Step 2. Governing Physics", "## Step 3. Generate Reference Solution")

        self.assertIn("we directly use the following two-state linearized swing-equation model", step_2.lower())
        self.assertEqual(step_2.count("$$"), 4)
        self.assertIn(r"\frac{d\delta}{dt}=\omega", step_2)
        self.assertIn(r"M\frac{d\omega}{dt} = \Delta P - D\omega - K_s\delta", step_2)

        expected_symbol_text = [
            r"$\delta(t)$",
            r"$\omega(t)$",
            r"$M$",
            r"$D$",
            r"$K_s$",
            r"$\Delta P$",
        ]
        for text in expected_symbol_text:
            self.assertIn(text, step_2)

        removed_derivation_text = [
            "P_m - P_e",
            r"P_e(\delta_0 + \Delta\delta)",
            "P_{max}",
            r"\left.\frac{dP_e}{d\delta}\right|_{\delta_0}",
        ]
        for text in removed_derivation_text:
            self.assertNotIn(text, step_2)

    def test_network_input_is_time_and_states_are_outputs(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")
        markdown = self._notebook_markdown().lower()
        code = self._notebook_code()

        self.assertIn("input is the time variable", markdown)
        self.assertIn("not the rotor angle", markdown)
        self.assertIn("outputs are", markdown)
        self.assertIn("delta", markdown)
        self.assertIn("omega", markdown)
        self.assertIn("input variables", markdown)
        self.assertIn("fixed-disturbance demo", markdown)
        self.assertIn("def forward(self, t):", code)
        self.assertIn("tau = 2.0 * t / self.t_end - 1.0", code)

    def test_sparse_measurement_design_has_enough_samples_for_this_demo(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")
        namespace = self._exec_setup_through_sparse_data()
        measurement_times = namespace["measurement_times"].reshape(-1)
        T_END = namespace["T_END"]

        self.assertGreaterEqual(measurement_times.size, 9)
        self.assertLessEqual(measurement_times.size, 15)
        self.assertAlmostEqual(float(measurement_times[0]), 0.0, places=12)
        self.assertAlmostEqual(float(measurement_times[-1]), float(T_END), places=12)
        self.assertTrue(np.all(np.diff(measurement_times) > 0.0))

    def test_notebook_enforces_pinn_accuracy_against_analytical_solution(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")
        code = self._notebook_code()

        expected_snippets = [
            "PINN_DELTA_MAE_TOLERANCE",
            "PINN_OMEGA_MAE_TOLERANCE",
            "assert pinn_delta_mae < PINN_DELTA_MAE_TOLERANCE",
            "assert pinn_omega_mae < PINN_OMEGA_MAE_TOLERANCE",
        ]

        for snippet in expected_snippets:
            self.assertIn(snippet, code)

    def test_step_9_compares_frequency_curves_and_reference_samples(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")
        code = self._notebook_code()

        expected_snippets = [
            "data_model = MLP(output_dim=2)",
            "data_only_pred = data_model(t_eval)",
            "data_only_delta_pred = data_only_pred[:, 0:1]",
            "data_only_omega_pred = data_only_pred[:, 1:2]",
            "data_only_omega_pred",
            "measurement_omega",
            "Synthetic omega samples",
            "Ordinary NN omega",
            "PINN omega",
        ]

        for snippet in expected_snippets:
            self.assertIn(snippet, code)

        self.assertNotIn("def predict_data_only_delta_and_omega", code)

    def test_data_only_baseline_uses_two_outputs_with_same_state_labels(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")
        code = self._notebook_code()
        notebook = self._load_notebook()
        markdown = "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "markdown"
        ).lower()

        expected_snippets = [
            "omega_data = torch.tensor(measurement_omega",
            "state_data = torch.cat([delta_data, omega_data], dim=1)",
            "initial_state",
            "pred_initial = model(t_initial)",
            "ic_loss = mse(pred_initial, initial_state)",
            "def train_data_only_model(model, t_data, state_data",
            "data_loss = mse(pred_data, state_data)",
            "data_only_history = train_data_only_model(data_model, t_data, state_data)",
        ]

        for snippet in expected_snippets:
            self.assertIn(snippet, code)

        self.assertIn("same two-output architecture", markdown)
        self.assertIn("same sparse synthetic rotor-angle and speed/frequency samples", markdown)
        self.assertIn("omega samples are used as training labels", markdown)

    def test_pinn_uses_same_state_labels_plus_physics_residual(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")
        code = self._notebook_code()
        markdown = self._notebook_markdown().lower()

        expected_snippets = [
            "pred_data = model(t_data)",
            "data_loss = mse(pred_data, state_data)",
            "physics_loss = mse(r_delta, torch.zeros_like(r_delta)) + mse(r_omega, torch.zeros_like(r_omega))",
            "total_loss = data_loss + lambda_physics * physics_loss + lambda_ic * ic_loss",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, code)

        self.assertIn("same sparse state samples", markdown)
        self.assertIn("plus the swing-equation residual", markdown)

    def test_ordinary_nn_training_process_is_visualized_and_explained(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")
        code = self._notebook_code()
        markdown = self._notebook_markdown().lower()

        expected_code_snippets = [
            "ax.semilogy(data_only_history[\"total_loss\"]",
            "ax.semilogy(data_only_history[\"data_loss\"]",
            "ax.semilogy(data_only_history[\"ic_loss\"]",
            "Ordinary NN Training Loss",
            "trained data_model",
            "data_model(t_eval)",
        ]
        for snippet in expected_code_snippets:
            self.assertIn(snippet, code)

        expected_markdown_snippets = [
            "ordinary nn training loss",
            "trained ordinary neural network",
            "evaluating the trained",
            "full time grid",
        ]
        for snippet in expected_markdown_snippets:
            self.assertIn(snippet, markdown)

    def test_problem_statement_matches_fair_two_output_comparison(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")
        notebook = self._load_notebook()
        markdown = "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "markdown"
        ).lower()

        self.assertNotIn("trained only on sparse synthetic rotor-angle samples", markdown)
        self.assertIn("same sparse", markdown)
        self.assertIn("same initial condition", markdown)
        self.assertIn("residual-free", markdown)
        self.assertIn("physics residual", markdown)

    def test_text_explicitly_marks_data_and_model_assumptions(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")

        notebook = self._load_notebook()
        markdown = "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "markdown"
        ).lower()

        self.assertIn("synthetic", markdown)
        self.assertIn("not field measurements", markdown)
        self.assertIn("linearized swing equation", markdown)
        self.assertIn("synchronizing power coefficient", markdown)
        self.assertIn("rotor angle", markdown)

    def test_references_support_swing_equation_and_pinn_sources(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")

        notebook = self._load_notebook()
        markdown = "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "markdown"
        )

        expected_reference_text = [
            "Power System Stability and Control",
            "10.1109/TPWRS.2004.825981",
            "Synchronizing Torque Impacts on Rotor Speed in Power Systems",
            "10.1109/TPWRS.2016.2600478",
            "Physics-informed neural networks",
            "10.1016/j.jcp.2018.10.045",
            "Physics-Informed Neural Networks for Power Systems",
            "1911.03737",
        ]

        for text in expected_reference_text:
            self.assertIn(text, markdown)

    def test_markdown_math_contains_no_json_control_escape_damage(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "PINN_Frequency.ipynb should exist")

        notebook = self._load_notebook()
        markdown = "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "markdown"
        )
        disallowed_controls = {
            "\f": "form feed, usually caused by writing \\frac as \\f + rac",
            "\r": "carriage return, usually caused by writing \\right as \\r + ight",
            "\t": "tab, often caused by writing \\text as \\t + ext",
        }

        for control_char, description in disallowed_controls.items():
            self.assertNotIn(control_char, markdown, description)

        self.assertIn(r"\frac{d\delta}{dt}", markdown)
        self.assertIn(r"\frac{d\omega}{dt}", markdown)
        self.assertIn(r"K_s", markdown)


if __name__ == "__main__":
    unittest.main()
