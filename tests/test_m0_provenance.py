"""Amendment 5 (A5.2 items 1 and 5): the M0 environment identity's recorded fields and the child's environment."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from sultai_screen.contracts import PilotHold
from sultai_screen.m0 import provenance as module
from sultai_screen.m0.provenance import ENVIRONMENT_VARIABLES, environment_identity, provenance

PINNED = {
    "torch_wheel": "2.9.1",
    "torch": "2.9.1+cu128",
    "cuda": "12.8",
    "numpy_wheel": "2.3.5",
    "numpy": "2.3.5",
}
RECORDED = {
    "python": "3.12.3",
    "driver": "580.65.06",
    "cudnn": 91002,
    "gpu_names": ["NVIDIA GeForce RTX 4060 Ti", "NVIDIA GeForce RTX 4060 Ti"],
}


class EnvironmentIdentityTests(unittest.TestCase):
    def patched(self, values: dict[str, str]) -> list[object]:
        query = "NVIDIA GeForce RTX 4060 Ti, 580.65.06\nNVIDIA GeForce RTX 4060 Ti, 580.65.06\n"
        return [
            patch(
                "sultai_screen.m0.provenance.importlib.metadata.version",
                side_effect=lambda name, values=values: values[name + "_wheel"],
            ),
            patch("sultai_screen.m0.provenance.torch.__version__", values["torch"]),
            patch("sultai_screen.m0.provenance.torch.version.cuda", values["cuda"]),
            patch("sultai_screen.m0.provenance.np.__version__", values["numpy"]),
            patch("sultai_screen.m0.provenance.platform.python_version", return_value="3.12.3"),
            patch("sultai_screen.m0.provenance.torch.backends.cudnn.version", return_value=91002),
            patch("sultai_screen.m0.provenance.subprocess.check_output", return_value=query),
            patch("sultai_screen.m0.provenance.torch.cuda.init"),
        ]

    def test_pinned_fields_refuse_a_mismatch_and_recorded_fields_never_do(self) -> None:
        # The five pins are unchanged; the four recorded fields (A5.2 item 1) are published, never compared.
        for bad in (None, *PINNED):
            values = dict(PINNED)
            if bad is not None:
                values[bad] = "wrong"
            patches = self.patched(values)
            for item in patches:
                item.start()
            try:
                if bad is None:
                    self.assertEqual(environment_identity(), {**PINNED, **RECORDED})
                else:
                    with self.assertRaisesRegex(PilotHold, bad):
                        environment_identity()
            finally:
                for item in patches:
                    item.stop()

    def test_an_unavailable_driver_query_or_cudnn_is_recorded_as_absent(self) -> None:
        patches = self.patched(dict(PINNED))
        patches[5] = patch("sultai_screen.m0.provenance.torch.backends.cudnn.version", return_value=None)
        patches[6] = patch("sultai_screen.m0.provenance.subprocess.check_output", side_effect=OSError("no nvidia-smi"))
        for item in patches:
            item.start()
        try:
            record = environment_identity()
        finally:
            for item in patches:
                item.stop()
        self.assertEqual((record["driver"], record["cudnn"], record["gpu_names"]), (None, None, None))
        self.assertEqual({key: record[key] for key in PINNED}, PINNED)
        self.assertEqual(set(module.PINNED_ENVIRONMENT), set(PINNED))


class ChildEnvironmentTests(unittest.TestCase):
    def test_provenance_records_the_whitelisted_variables_and_names_absent_ones(self) -> None:
        # A5.2 item 5: the child's own receipt carries the determinism-relevant variables it ran under.
        self.assertEqual(
            ENVIRONMENT_VARIABLES,
            (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
                "CUDA_LAUNCH_BLOCKING",
                "PYTORCH_CUDA_ALLOC_CONF",
                "TORCH_CUDNN_V8_API_DISABLED",
            ),
        )
        present = {"CUDA_VISIBLE_DEVICES": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2"}
        with (
            patch.dict("sultai_screen.m0.provenance.os.environ", present, clear=True),
            patch("sultai_screen.m0.provenance.source_identity", return_value={"commit": "c" * 40}),
            patch("sultai_screen.m0.provenance.environment_identity", return_value={"torch": "fake"}),
            patch("sultai_screen.m0.provenance.flags", return_value={"deterministic": True}),
        ):
            record = provenance()
        variables = record["environment_variables"]
        assert isinstance(variables, dict)
        self.assertEqual(list(variables), list(ENVIRONMENT_VARIABLES))
        self.assertEqual({k: v for k, v in variables.items() if v is not None}, present)
        self.assertEqual(record["physical_gpu"], "1")
        self.assertEqual(record["determinism"], {"deterministic": True})
