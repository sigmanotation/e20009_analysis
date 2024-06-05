from attpc_engine.kinematics import (
    KinematicsPipeline,
    KinematicsTargetMaterial,
    ExcitationGaussian,
    ExcitationUniform,
    run_kinematics_pipeline,
    Reaction,
    Decay,
)
from attpc_engine.detector import (
    DetectorParams,
    ElectronicsParams,
    PadParams,
    Config,
    run_simulation,
)
from attpc_engine.detector.response import get_response
from attpc_engine.detector.writer import convert_to_spyral
from attpc_engine import nuclear_map
from spyral_utils.nuclear.target import TargetData, GasTarget
from pathlib import Path

import h5py as h5
import numpy as np

#############################################################################################


# For experiment e20009 we need to create a unique writer because we append IC SCA
# information to the point clouds
class SpyralWriter_e20009:

    def __init__(self, file_path: Path, config: Config):
        self.path = file_path
        self.file = h5.File(self.path, "w")
        self.cloud_group = self.file.create_group("cloud")
        self.response = get_response(config).copy()

    def write(self, data: np.ndarray, config: Config, event_number: int) -> None:
        if config.pad_centers is None:
            raise ValueError("Pad centers are not assigned at write!")
        spyral_format = convert_to_spyral(
            data,
            config.elec_params.windows_edge,
            config.elec_params.micromegas_edge,
            config.det_params.length,
            self.response,
            config.pad_centers,
            config.elec_params.adc_threshold,
        )

        dset = self.cloud_group.create_dataset(
            f"cloud_{event_number}", data=spyral_format
        )
        # No ic stuff from simulation
        dset.attrs["ic_amplitude"] = -1.0
        dset.attrs["ic_multiplicity"] = -1.0
        dset.attrs["ic_integral"] = -1.0
        dset.attrs["ic_centroid"] = -1.0

        # This is needed for experiment e20009
        dset.attrs["ic_sca_centroid"] = -1.0
        dset.attrs["ic_sca_multiplicity"] = -1.0

    def set_number_of_events(self, n_events: int) -> None:
        self.cloud_group.attrs["min_event"] = 0
        self.cloud_group.attrs["max_event"] = n_events - 1

    def get_filename(self) -> Path:
        return self.path


# Used to determine the maximum energy for ExcitationUniform
def max_reaction_excitation_energy(reaction: dict, beam_energy: float):
    """
    Calculates the max excitation energy in MeV that can be given to a single nucleus in the exit
    channel of the nuclear reaction A + B -> C + D + F ... by solving the relativistic
    minimum beam kinetic energy condition. See https://web.physics.utah.edu/~jui/5110/hw/kin_rel.

    Parameters
    ----------
    reaction: dict[spyral_utils.nuclear.nuclear_map.NucleusData]
        Dictionary of reaction nuclei. The keys are "beam", "target", and "products".
        The beam and target keys hold one NucleusData instance, while the products key
        is a list of NucleusData instances, one for each product.
    beam_energy: float
        Beam kinetic energy of beam nucleus in MeV.
    """
    # Calculate sum of all rest masses of exit channel nuclei
    M = 0
    for nucleus in reaction["products"]:
        M += nucleus.mass

    # Quadratic formula variables
    a = 1
    b = 2 * M
    c = (
        M**2
        - (reaction["beam"].mass + reaction["target"].mass) ** 2
        - 2 * reaction["target"].mass * beam_energy
    )

    # Maximum excitation energy
    ex = (-b + np.sqrt(b**2 - 4 * a * c)) / 2 / a

    return ex


#############################################################################################
# Set output file paths for simulated kinematic events and point clouds
kine_path = Path(
    "/Users/attpc/Desktop/e20009_analysis/e20009_analysis/simulations/breakup_kine.h5"
)
det_path = Path(
    "/Users/attpc/Desktop/e20009_analysis/e20009_analysis/simulations/run_0001.h5"
)

target = GasTarget(
    TargetData(compound=[(1, 2, 2)], pressure=600.0, thickness=None), nuclear_map
)

nevents = 500000

# At least 2.22 MeV is needed to break up the deuteron.
pipeline = KinematicsPipeline(
    [
        Reaction(
            target=nuclear_map.get_data(1, 2),
            projectile=nuclear_map.get_data(4, 10),
            ejectile=nuclear_map.get_data(4, 10),
        ),
        Decay(parent=nuclear_map.get_data(1, 2), residual_1=nuclear_map.get_data(1, 1)),
    ],
    [ExcitationUniform(2.2, 13.9), ExcitationGaussian(0.0)],
    beam_energy=93.0,  # MeV
    target_material=KinematicsTargetMaterial(
        material=target, z_range=(0.0, 1.0), rho_sigma=0.02 / 3
    ),
)

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


#############################################################################################
def main():
    # run_kinematics_pipeline(pipeline, nevents, kine_path)
    run_simulation(
        config,
        kine_path,
        writer,
    )

    # reaction = {
    #     "beam": nuclear_map.get_data(4, 10),
    #     "target": nuclear_map.get_data(1, 2),
    #     "products": [nuclear_map.get_data(4, 11), nuclear_map.get_data(1, 1)],
    # }

    # print(max_reaction_excitation_energy(reaction, 93))


if __name__ == "__main__":
    main()
