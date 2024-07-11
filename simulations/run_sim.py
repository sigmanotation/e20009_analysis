from attpc_engine.kinematics import (
    KinematicsPipeline,
    KinematicsTargetMaterial,
    ExcitationGaussian,
    ExcitationUniform,
    ExcitationBreitWigner,
    run_kinematics_pipeline,
    Reaction,
    Decay,
    PolarUniform,
)
from attpc_engine.detector import (
    DetectorParams,
    ElectronicsParams,
    PadParams,
    Config,
    run_simulation,
)
from attpc_engine import nuclear_map
from spyral_utils.nuclear.target import TargetData, GasTarget
from sim_utils import SpyralWriter_e20009, max_reaction_excitation_energy

from pathlib import Path
import numpy as np


# Set output file paths for simulated kinematic events and point clouds
kine_path = Path("/Volumes/e20009_sim/deuteron_bu_0-60/deut_bu_kine.h5")
det_path = Path("/Volumes/e20009_sim/deuteron_bu_0-60/")

target = GasTarget(
    TargetData(compound=[(1, 2, 2)], pressure=600.0, thickness=None), nuclear_map
)

# Number of events to simulate
nevents = 500000

# Deuteron break up pipeline
# At least 2.22 MeV is needed to break up the deuteron.
# Remember, the min and max angles here are the cm angles for the proton
# and we plot angular distributions wrt to the heavy residue, so subtract 180
pipeline = KinematicsPipeline(
    [
        Reaction(
            target=nuclear_map.get_data(1, 2),
            projectile=nuclear_map.get_data(4, 10),
            ejectile=nuclear_map.get_data(4, 10),
        ),
        Decay(parent=nuclear_map.get_data(1, 2), residual_1=nuclear_map.get_data(1, 1)),
    ],
    [
        ExcitationUniform(2.2, 15.58),
        ExcitationGaussian(0.0),
    ],
    [
        PolarUniform(angle_min=0, angle_max=np.pi),
        PolarUniform(angle_min=(2 * np.pi / 3), angle_max=np.pi),
    ],
    beam_energy=93.0,  # MeV
    target_material=KinematicsTargetMaterial(
        material=target, z_range=(0.0, 1.0), rho_sigma=0.02 / 3
    ),
)

# 10Be(d,p) pipeline
# pipeline = KinematicsPipeline(
#     [
#         Reaction(
#             target=nuclear_map.get_data(1, 2),
#             projectile=nuclear_map.get_data(4, 10),
#             ejectile=nuclear_map.get_data(1, 1),
#         )
#     ],
#     [
#         ExcitationBreitWigner(
#             rest_mass=nuclear_map.get_data(4, 11).mass,
#             centroid=1.78,
#             width=0.100,
#         )
#     ],
#     [PolarUniform(angle_min=(2 * np.pi / 3), angle_max=np.pi)],
#     beam_energy=93.0,  # MeV
#     target_material=KinematicsTargetMaterial(
#         material=target, z_range=(0.0, 1.0), rho_sigma=0.02 / 3
#     ),
# )

detector = DetectorParams(
    length=1.0,
    efield=60000.0,
    bfield=3.0,
    mpgd_gain=175000,
    gas_target=target,
    diffusion=0.277,
    fano_factor=0.2,
    w_value=34.0,
)

electronics = ElectronicsParams(
    clock_freq=3.125,
    amp_gain=900,
    shaping_time=1000,
    micromegas_edge=60,
    windows_edge=400,
    adc_threshold=30.0,
)

pads = PadParams()

config = Config(detector, electronics, pads)
writer = SpyralWriter_e20009(det_path, config)


def main():
    run_kinematics_pipeline(pipeline, nevents, kine_path)
    run_simulation(
        config,
        kine_path,
        writer,
    )

    # reaction = {
    #     "beam": nuclear_map.get_data(4, 10),
    #     "target": nuclear_map.get_data(1, 2),
    #     "products": [nuclear_map.get_data(4, 10), nuclear_map.get_data(1, 2)],
    # }

    # print(max_reaction_excitation_energy(reaction, 93))


if __name__ == "__main__":
    main()
