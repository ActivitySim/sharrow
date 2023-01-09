import numpy as np
import pytest
import yaml

pytest.importorskip("larch.model.tree")


def test_nesting_arrays():
    nest_yaml = """
    NESTS:
      name: root
      coefficient: coef_nest_root
      alternatives:
          - name: AUTO
            coefficient: coef_nest_AUTO
            alternatives:
                - name: DRIVEALONE
                  coefficient: coef_nest_AUTO_DRIVEALONE
                  alternatives:
                    - DRIVEALONEFREE
                    - DRIVEALONEPAY
                - name: SHAREDRIDE2
                  coefficient: coef_nest_AUTO_SHAREDRIDE2
                  alternatives:
                    - SHARED2FREE
                    - SHARED2PAY
                - name: SHAREDRIDE3
                  coefficient: coef_nest_AUTO_SHAREDRIDE3
                  alternatives:
                    - SHARED3FREE
                    - SHARED3PAY
          - name: NONMOTORIZED
            coefficient: coef_nest_NONMOTORIZED
            alternatives:
              - WALK
              - BIKE
          - name: TRANSIT
            coefficient:  coef_nest_TRANSIT
            alternatives:
                - name: WALKACCESS
                  coefficient: coef_nest_TRANSIT_WALKACCESS
                  alternatives:
                  - WALK_LOC
                  - WALK_LRF
                  - WALK_EXP
                  - WALK_HVY
                  - WALK_COM
                - name: DRIVEACCESS
                  coefficient: coef_nest_TRANSIT_DRIVEACCESS
                  alternatives:
                  - DRIVE_LOC
                  - DRIVE_LRF
                  - DRIVE_EXP
                  - DRIVE_HVY
                  - DRIVE_COM
          - name: RIDEHAIL
            coefficient: coef_nest_RIDEHAIL
            alternatives:
              - TAXI
              - TNC_SINGLE
              - TNC_SHARED
    """
    nests = yaml.safe_load(nest_yaml)["NESTS"]
    alternatives = [
        "DRIVEALONEFREE",
        "DRIVEALONEPAY",
        "SHARED2FREE",
        "SHARED2PAY",
        "SHARED3FREE",
        "SHARED3PAY",
        "WALK",
        "BIKE",
        "WALK_LOC",
        "WALK_LRF",
        "WALK_EXP",
        "WALK_HVY",
        "WALK_COM",
        "DRIVE_LOC",
        "DRIVE_LRF",
        "DRIVE_EXP",
        "DRIVE_HVY",
        "DRIVE_COM",
        "TAXI",
        "TNC_SINGLE",
        "TNC_SHARED",
    ]
    from ..nested_logit import construct_nesting_tree

    tree = construct_nesting_tree(alternatives, nests)
    arrs = tree.as_arrays(
        trim=True,
        parameter_dict={
            "coef_nest_AUTO_DRIVEALONE": 0.3,
            "coef_nest_AUTO_SHAREDRIDE2": 0.4,
            "coef_nest_AUTO_SHAREDRIDE3": 0.4,
            "coef_nest_AUTO": 0.6,
            "coef_nest_NONMOTORIZED": 0.72,
            "coef_nest_TRANSIT": 0.72,
            "coef_nest_TRANSIT_WALKACCESS": 0.5,
            "coef_nest_TRANSIT_DRIVEACCESS": 0.5,
            "coef_nest_RIDEHAIL": 0.36,
        },
    )
    expected = {
        "n_nodes": 31,
        "n_alts": 21,
        "edges_up": np.array(
            [
                21,
                21,
                22,
                22,
                23,
                23,
                24,
                24,
                24,
                25,
                25,
                26,
                26,
                26,
                26,
                26,
                27,
                27,
                27,
                27,
                27,
                28,
                28,
                29,
                29,
                29,
                30,
                30,
                30,
                30,
            ],
            dtype=np.int32,
        ),
        "edges_dn": np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                21,
                22,
                23,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                26,
                27,
                18,
                19,
                20,
                24,
                25,
                28,
                29,
            ],
            dtype=np.int32,
        ),
        "edges_1st": np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            dtype=np.int32,
        ),
        "edges_alloc": np.array(
            [
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
            dtype=np.int32,
        ),
        "mu_params": np.array(
            [0.3, 0.4, 0.4, 0.6, 0.72, 0.5, 0.5, 0.72, 0.36, 1.0], dtype=np.float32
        ),
        "start_slots": np.array([0, 2, 4, 6, 9, 11, 16, 21, 23, 26], dtype=np.int32),
        "len_slots": np.array([2, 2, 2, 3, 2, 5, 5, 2, 3, 4], dtype=np.int32),
    }
    for k, v in expected.items():
        assert arrs[k] == pytest.approx(v), f"bad {k!r}"
