import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm
import os

n_epochs = 20000
skip_steps = 10

Re = 60
Nx = 300
Ny = 100
e = 4
Cx = Nx//5
Cy = Ny//2
R = Ny//5
u = 0.04

N_DISCRETE_VELOCITIES = 9

LATTICE_VELOCITIES = np.array([
    [ 0,  1,  0, -1,  0,  1, -1, -1,  1,],
    [ 0,  0,  1,  0, -1,  1,  1, -1, -1,]
])

LATTICE_INDICES = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8,
])

OPPOSITE_LATTICE_INDICES = np.array([
    0, 3, 4, 1, 2, 7, 8, 5, 6,
])

LATTICE_WEIGHTS = np.array([
    4/9,
    1/9,  1/9,  1/9,  1/9,
    1/36, 1/36, 1/36, 1/36,
])

RIGHT_VELOCITIES = np.array([1, 5, 8])
UP_VELOCITIES = np.array([2, 5, 6])
LEFT_VELOCITIES = np.array([3, 6, 7])
DOWN_VELOCITIES = np.array([4, 7, 8])
PURE_VERTICAL_VELOCITIES = np.array([0, 2, 4])
PURE_HORIZONTAL_VELOCITIES = np.array([0, 1, 3])

def Cpt(θ):
    return 1.0 - 4.0 * (np.sin(θ * np.pi / 180.0))**2

def get_density(discrete_velocities):
    density = np.sum(discrete_velocities, axis=-1)
    return density

def get_macroscopic_velocities(discrete_velocities, density):
    macroscopic_velocities = np.einsum(
        "NMQ,dQ->NMd",
        discrete_velocities,
        LATTICE_VELOCITIES,
    ) / density[..., np.newaxis]

    return macroscopic_velocities

def get_equilibrium_discrete_velocities(macroscopic_velocities, density):
    projected_discrete_velocities = np.einsum(
        "dQ,NMd->NMQ",
        LATTICE_VELOCITIES,
        macroscopic_velocities,
    )
    macroscopic_velocity_magnitude = np.linalg.norm(
        macroscopic_velocities,
        axis=-1,
        ord=2,
    )
    equilibrium_discrete_velocities = (
        density[..., np.newaxis]
        *
        LATTICE_WEIGHTS[np.newaxis, np.newaxis, :]
        *
        (
            1
            +
            3 * projected_discrete_velocities
            +
            9/2 * projected_discrete_velocities**2
            -
            3/2 * macroscopic_velocity_magnitude[..., np.newaxis]**2
        )
    )

    return equilibrium_discrete_velocities

def main():

    if not os.path.exists("images"):
        os.mkdir("images")

    # Kinematic Viscosity
    v = u*R/Re

    # Relaxation Factor for Lattice Boltzman iteration
    ω = 1.0 / (3.0 * v + 0.5)

    # Creating a 2d mesh
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    object_mask = np.sqrt(( X - Cx )**2 + ( Y - Cy )**2) < R

    velocity_profile = np.zeros((Nx, Ny, 2))
    velocity_profile[:, :, 0] = u

    def update(discrete_velocities_prev):
        # (1) Prescribe the outflow BC on the right boundary
        discrete_velocities_prev[-1, :, LEFT_VELOCITIES] = discrete_velocities_prev[-2, :, LEFT_VELOCITIES]

        # (2) Macroscopic Velocities
        density_prev = get_density(discrete_velocities_prev)
        macroscopic_velocities_prev = get_macroscopic_velocities(
            discrete_velocities_prev,
            density_prev,
        )

        # (3) Prescribe Inflow Dirichlet BC using Zou/He scheme
        macroscopic_velocities_prev[0, 1:-1, :] = velocity_profile[0, 1:-1, :]
        density_prev[0, :] = \
            (
                get_density(discrete_velocities_prev[0, :, PURE_VERTICAL_VELOCITIES].T)
                +
                2 *
                get_density(discrete_velocities_prev[0, :, LEFT_VELOCITIES].T)
            ) / (
                1 - macroscopic_velocities_prev[0, :, 0]
            )

        # (4) Compute discrete Equilibria velocities
        equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(
            macroscopic_velocities_prev,
            density_prev,
        )

        # (3) Belongs to the Zou/He scheme
        discrete_velocities_prev[0, :, RIGHT_VELOCITIES] = equilibrium_discrete_velocities[0, :, RIGHT_VELOCITIES]
        
        # (5) Collide according to BGK
        discrete_velocities_post_collision = (
            discrete_velocities_prev - ω *(discrete_velocities_prev-equilibrium_discrete_velocities)
        )

        # (6) Bounce-Back Boundary Conditions to enfore the no-slip
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_post_collision[object_mask, LATTICE_INDICES[i]] = discrete_velocities_prev[object_mask, OPPOSITE_LATTICE_INDICES[i]]
        
        # (7) Stream alongside lattice velocities
        discrete_velocities_streamed = discrete_velocities_post_collision
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_streamed[:, :, i] = \
                np.roll(
                    np.roll(
                        discrete_velocities_post_collision[:, :, i],
                        LATTICE_VELOCITIES[0, i],
                        axis=0,
                    ),
                    LATTICE_VELOCITIES[1, i],
                    axis=1,
                )
        
        return discrete_velocities_streamed

    discrete_velocities_prev = get_equilibrium_discrete_velocities(velocity_profile, np.ones((Nx, Ny)))
    θ = np.linspace(0, 180, 1000)
    x = np.array((R+e)*np.cos(θ * np.pi / 180.0 - np.pi) + Cx, dtype=np.int32)
    y = np.array((R+e)*np.sin(θ * np.pi /180.0 - np.pi) + Cy, dtype=np.int32)
    plt.figure(figsize=(15, 6), dpi=120)

    for idx in tqdm(range(n_epochs)):
        
        discrete_velocities_next = update(discrete_velocities_prev)
        discrete_velocities_prev = discrete_velocities_next

        if idx % skip_steps == 0:

            density = get_density(discrete_velocities_next)
            macroscopic_velocities = get_macroscopic_velocities(discrete_velocities_next, density)
            velocity_magnitude = np.linalg.norm(macroscopic_velocities, axis = -1, ord = 2)

            d_u__d_x, d_u__d_y = np.gradient(macroscopic_velocities[..., 0])
            d_v__d_x, d_v__d_y = np.gradient(macroscopic_velocities[..., 1])
            curl = (d_u__d_y - d_v__d_x)

            plt.subplot(221)
            plt.contourf(X, Y, velocity_magnitude, levels=200)
            plt.colorbar().set_label("Velocity Magnitude")
            plt.gca().add_patch(plt.Circle((Cx, Cy), R, color="darkred"))
            plt.text(0, 110, f"Iteration : {idx}", fontsize=16)

            plt.subplot(223)
            plt.contourf(X, Y, curl, levels=200, cmap=cmr.fusion, vmin=-0.02, vmax=0.02)
            plt.colorbar().set_label("Vorticity Magnitude")
            plt.gca().add_patch(plt.Circle((Cx, Cy), R, color="darkgreen"))

            vel_mag = velocity_magnitude[x, y]
            density_value = density[x, y]

            plt.subplot(222)
            plt.plot(θ, vel_mag)
            plt.xlabel("Theta")
            plt.ylabel("Velocity Magnitude")

            plt.subplot(224)
            plt.plot(θ, Cpt(θ), label="Theoretical")
            plt.plot(θ, 1 - 0.5*(density_value * vel_mag/u)**2, label="Experimental")
            plt.legend()
            plt.xlabel("Theta")
            plt.ylabel("Cp")

            plt.savefig(f"images/{idx//skip_steps}.png")
            plt.clf()

if __name__ == "__main__":
    main()