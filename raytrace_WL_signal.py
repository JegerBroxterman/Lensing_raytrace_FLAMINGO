# Ray tracing the weak lensing signal back to multiple HEALPix source planes weighted by a source distribution 
# For a detailed explantion of what the code does see https://ui.adsabs.harvard.edu/abs/2024MNRAS.529.2309B/abstract

# The code requires:
# 1. precomputed overdensity HEALPix shells that are loaded in delta()
# 2. The inner and outer comoving distances (in Mpc) of each shell in get_shells_properties()
# 3. A few quantities need to be set at the beginning; the HEALPix N_SIDE, number of shells, an astropy cosmology instance of the simulation, name of the output file

# 4. You can choose between NGP (nearest gridpoint) or bilinear interpolation by setting NGP = True or False, respectively. This sets the method of determinging the deflection angle and shear matrix at each shell. NGP uses the value of the closest pixel center whereas bilinear interpolation is bilinear interpolation of the 4 closest pixels.
# 5. You need to decide whether you want to define the centers of the shells as the centers in redshift or volume, by setting centre_in_redshift = True or False, respectively
# 6. You need to choose if you want a redshift distribution or not. You can set your redshift distribution in the function n_chi_ana() and by setting redshift_distribution = True. If redshift_distribution = False, it defaults to a Dirac delta function at the final shell.

import os
import numpy as np
import healpy as hp
import unyt
import astropy.units as u
from astropy.cosmology import z_at_value
import scipy.integrate as integrate
from astropy.cosmology import w0waCDM
from tqdm import tqdm
import h5py
import lightcone_io.healpix_maps as hm
import swiftsimio as sw

unyt.c.convert_to_units(unyt.km / unyt.s)

#FLAMINGO initialisation
variation = "DMO_FIDUCIAL"
name="L2p8m9_DMOF"
thebox = f"L1000N1800/{variation}/"
path = f"/cosma8/data/dp004/flamingo/Runs/{thebox}/neutrino_corrected_maps_downsampled_4096/" 
snap = sw.load(f"/cosma8/data/dp004/flamingo/Runs/{thebox}/snapshots/flamingo_0077/flamingo_0077.hdf5") #To read in the cosmology 
shells = hm.ShellArray(f"{path}", "lightcone0")
cosmo = snap.metadata.cosmology 

NGP = True  # True for determining the quantaties at each shell using the nearest gridpoint, False for bilinear interpolation
redshift_distribution = True  # False for n(z) = delta(z=z_final_shell). True for distribution set in n_chi_ana()
centre_in_redshift = True  # True of setting the centre of the shell as (z_max+z_min)/2 (for each shell), False for centring in volume such that Volume(inner_boundary to centre) = Volume(centre to outer boundary)

output_conv = "convergence_file_z3_raytraced.hdf5"  # output file name of the final convergence map
NR_shells = shells.nr_shells  # the number of shells in the
#cosmo = w0waCDM(H0=68.1,Om0=0.30461,Tcmb0=2.7255,Ob0=0.0486,Ode0=0.694,Neff=3.044,w0=-1.0,wa=0.0)  # An astropy cosmology instance
nside = 1024  # HEALPix Nside
npix = int(12 * nside ** 2)
lrange, mrange = hp.Alm.getlm(lmax=3*nside-1)
theta_pixels,phi_pixels = hp.pixelfunc.pix2ang(nside,range(npix))
conv = np.zeros((npix))  # the final convergence map

delta_chis = np.zeros(NR_shells)  # Thickness of the shells
mid_zs = np.zeros(NR_shells)  # Central redshifts
comoving_distances = np.zeros(NR_shells) * unyt.Mpc  # Central comoving distances
zs_volume_weighted = np.zeros(NR_shells)  # Redshift according to volume weighting
chi_volume_weighted = (np.zeros(NR_shells) * unyt.Mpc)  # Comoving distance according to volume weighting
r_mins = np.zeros(NR_shells) * unyt.Mpc  # Inner comoving distance
r_maxs = np.zeros(NR_shells) * unyt.Mpc  # Outer comoving distance


def get_shells_properties():
    for ishell in range(NR_shells):
        rmin =  shells[ishell].comoving_inner_radius # Inner comoving distance of shell in Mpc
        rmax =  shells[ishell].comoving_outer_radius # Outer comoving distance of shell in Mpc
        r_mins[ishell] = rmin
        r_maxs[ishell] = rmax
        delta_chis[ishell] = rmax - rmin
        comoving_distances[ishell] = (rmin + rmax) / 2  # Value in Mpc
        chi_volume_weighted[ishell] = ((rmax ** 3 + rmin ** 3) / 2) ** (1.0 / 3.0)
        z_mid = z_at_value(
            cosmo.comoving_distance, (rmax.value + rmin.value) / 2 * u.Mpc
        )
        mid_zs[ishell] = z_mid
        zs_volume_weighted[ishell] = z_at_value(
            cosmo.comoving_distance,
            ((rmax.value ** 3 + rmin.value ** 3) / 2) ** (1.0 / 3.0) * u.Mpc,
        )
    return


get_shells_properties()


if centre_in_redshift:
    plane_distances = comoving_distances
    redshifts = mid_zs

else:
    plane_distances = chi_volume_weighted
    redshifts = zs_volume_weighted

# When using an analytic redshift distribution
def n_chi_ana(co_dists):
    zm = 0.9
    z0 = zm / np.sqrt(2)
    counts_z = (redshifts / z0) ** 2 * np.exp(
        -((redshifts / z0) ** (3 / 2))
    )  # Simple Euclid forcast -> Blanchard et al. 2020
    counts_chi = counts_z * cosmo.H(redshifts).value / unyt.c.value
    return counts_chi / integrate.simpson(counts_chi,x= co_dists)

def n_chi_delta(co_dists):
    zs = redshifts
    counts_z = np.zeros((len(redshifts)))
    counts_z[10] = 1.
    counts_chi = counts_z*cosmo.H(zs).value/unyt.c.value
    return counts_chi/integrate.simpson(counts_chi,x=co_dists)

n_chi = n_chi_ana(plane_distances)  # source comoving redshift distribution

def downgrade(old_map,nside_new):
	return hp.pixelfunc.ud_grade(old_map, nside_new) #-2 keeps sum invariant, 0 the mean 
	
def rotate_map(hmap, rot_theta, rot_phi):
    longitude = rot_phi*180/np.pi * u.deg
    latitude = rot_theta*180/np.pi * u.deg
    if longitude.value != 0. and latitude.value != 0.:
        rot_custom = hp.Rotator(rot=[longitude.to_value(u.deg), latitude.to_value(u.deg)], inv=True)
        hmap = rot_custom.rotate_map_alms(hmap)
    return hmap	 
	
# 2D overdensity map for ith shell.
def delta(i):
        data_file = h5py.File(f"/cosma8/data/dp004/jch/Jeger/HYDRO_FIDUCIAL/Rotated_overdensity_map_HF_L1m9_shell{i}_lc0.hdf5",'r')
        overdens = downgrade(data_file["Overdensity"][:],nside)  
        #overdens = downgrade(overdens,nside_new) #If you want to downgrade the map
       # overdens = rotate_map(overdens, rot_theta, rot_phi) #If you want to rotate the map
        return overdens

def conv_of_ith_shell(i):
    conv_value = (
        3 / 2
        * cosmo.Om0
        * cosmo.H0.value ** 2
        / unyt.c.value ** 2
        * plane_distances[i]
        * (1 + redshifts[i])
        * delta(i)
        * delta_chis[i]
    )
    return conv_value.value

def shear_matrix_and_deflection_field_at_i(i):
    """
    Determines the deflection field and shear matrix/ tidal tensor/ lensing Jacobian
    of the i-th plane by determining the (first and second order, respectively) covariant
    derivates of the lensing potential.
    """
    convergence = conv_of_ith_shell(i)  # Convergence field for i-th plane
    K_lm = hp.map2alm(convergence)  # Spherical harmonics of the conv field

    psi_lm = -K_lm * 2 / lrange / (lrange + 1)  # A warning may occur as the first coefficient is undefined 
    psi_lm[0] = 0 # It is set manually to zero after

    _, dtheta, dphi = hp.alm2map_der1(psi_lm, nside)  # healpy already scales dphi by sin(theta)

    dtheta_lm = hp.map2alm(dtheta)
    dphi_lm = hp.map2alm(dphi)

    _, dthetadtheta, dthetadphi = hp.alm2map_der1(dtheta_lm, nside)
    _, _, dphidphi = hp.alm2map_der1(dphi_lm, nside)  # All second order partial derivatives
    return np.array(
        (
            [
                dthetadtheta,
                dthetadphi - np.cos(theta_pixels) / np.sin(theta_pixels) * dphi,
                dphidphi + np.cos(theta_pixels) / np.sin(theta_pixels) * dtheta,
            ]
        )
    ), np.array(([dtheta, dphi]))
# Minus conventions consistent with the recurrance relations from Hilbert2009

def transport_mag_matrix_NGP(positions, mag_matrix):
    """
    Parallel transport the magnification matrix of the NGP pixel along the geodesic connecting
    the NGP centres to the original ray postition. This is done by rotating the coordinate system
    such that both the initial and NGP position lay on the equator, in which case the tensor
    components remain constant as it is transported along the geodesic connecting the centres. 
    See also Becker+2013 (CALCLENS) appendix A
    """
    unit_phi = np.array(([-pixels_cart[1, :], pixels_cart[0, :], np.zeros((npix))]))
    unit_theta = np.array(
        (
            [
                pixels_cart[0, :] * pixels_cart[2, :],
                pixels_cart[1, :] * pixels_cart[2, :],
                -(pixels_cart[0, :] ** 2 + pixels_cart[1, :] ** 2),
            ]
        )
    )

    cross = np.cross(positions, pixels_cart, axis=0)
    cos = (positions * pixels_cart).sum(axis=0)  # a.b = |a||b|cos
    sin = np.linalg.norm(cross, axis=0)  # axb=|a||b|sin
    cross = cross / sin
    angle_mask = sin == 0. 
    cross[0,:][angle_mask] = 1.
    cross[1,:][angle_mask] = 0.
    cross[2,:][angle_mask] = 0.

    rotated_basis = np.array(([-positions[1, :], positions[0, :], np.zeros((npix))]))
    dot_product = (rotated_basis * cross).sum(axis=0)
    cross_product = np.cross(cross, rotated_basis, axis=0)
    rot_unit_phi_n = (
        rotated_basis * cos + cross * dot_product * (1 - cos) + cross_product * sin
    )

    sin_angle = (rot_unit_phi_n * unit_theta).sum(axis=0) / np.sqrt(
        (1.0 - pixels_cart[2, :] ** 2) * (1.0 - positions[2, :] ** 2)
    )
    cos_angle = (rot_unit_phi_n * unit_phi).sum(axis=0) / np.sqrt(
        (1.0 - pixels_cart[2, :] ** 2) * (1.0 - positions[2, :] ** 2)
    )
    rotmat = np.array(([cos_angle, -sin_angle], [sin_angle, cos_angle]))
    return np.einsum(
        "jil,jkl->ikl", rotmat, np.einsum("ijl,jkl->ikl", mag_matrix, rotmat)
    )

def transport_mag_matrices_neighbours(neigbour_positions, mag_matrix):
    """
    Parallel transport the magnification matrix of the 4 neighbouring pixel along the geodesic connecting
    the neigbhour centres to the original ray postition. This is done by rotating the coordinate system
    such that both the initial and neighbour position lay on the equator, in which case the tensor
    components remain constant as it is transported along the sphere (the Chirstoffels vanish), see 
    Becker+2013 CALCLENS appendix A.
    """
    cross = np.cross(
        neigbour_positions, np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1), axis=0
    )
    sin = np.linalg.norm(cross, axis=0)  # |axb|=|a||b||sin|
    cross = cross / sin
    angle_mask = sin == 0
    cross[0,:,:][angle_mask] = 1
    cross[1,:,:][angle_mask] = 0
    cross[2,:,:][angle_mask] = 0

    rot_unit_phi_n = (
        np.array(
            (
                [
                    -neigbour_positions[1, :, :],
                    neigbour_positions[0, :, :],
                    np.zeros((4, npix)),
                ]
            )
        )
        * (
            neigbour_positions * np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1)
        ).sum(axis=0)
        + cross
        * (
            np.array(
                (
                    [
                        -neigbour_positions[1, :, :],
                        neigbour_positions[0, :, :],
                        np.zeros((4, npix)),
                    ]
                )
            )
            * cross
        ).sum(axis=0)
        * (
            1
            - (
                neigbour_positions * np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1)
            ).sum(axis=0)
        )
        + np.cross(
            cross,
            np.array(
                (
                    [
                        -neigbour_positions[1, :, :],
                        neigbour_positions[0, :, :],
                        np.zeros((4, npix)),
                    ]
                )
            ),
            axis=0,
        )
        * sin
    )

    sin_angle = (
        rot_unit_phi_n
        * np.array(
            (
                [
                    np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1)[0, :, :]
                    * np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1)[2, :, :],
                    np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1)[1, :, :]
                    * np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1)[2, :, :],
                    -(
                        np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1)[0, :, :]
                        ** 2
                        + np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1)[1, :, :]
                        ** 2
                    ),
                ]
            )
        )
    ).sum(axis=0) / np.sqrt(
        (1.0 - np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1)[2, :, :] ** 2)
        * (1.0 - neigbour_positions[2, :, :] ** 2)
    )
    cos_angle = (
        rot_unit_phi_n
        * np.array(
            (
                [
                    -np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1)[1, :, :],
                    np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1)[0, :, :],
                    np.zeros((4, npix)),
                ]
            )
        )
    ).sum(axis=0) / np.sqrt(
        (1.0 - np.repeat(pixels_cart[:, np.newaxis, :], 4, axis=1)[2, :, :] ** 2)
        * (1.0 - neigbour_positions[2, :, :] ** 2)
    )
    rot_unit_phi_n, unit_phi = 0.0, 0.0  
    rotmat = np.array(
        ([cos_angle, -sin_angle], [sin_angle, cos_angle])
    )  
    return np.einsum(
        "jiml,jkml->ikml", rotmat, np.einsum("ijml,jkml->ikml", mag_matrix, rotmat)
    )


pixels_cart = np.array(hp.pixelfunc.pix2vec(nside, range(npix), nest=False))  # The xyz of all the pixels

A_initial = np.array(([1, 0], [0, 1]))  # The initial maginification matrix
mag_matrix = np.zeros((2, 2, 3, npix))  # Magnification matrix (2x2) for all rays (npix) at three (3) planes
mag_matrix[:, :, 0, :] = np.repeat(A_initial[:, :, np.newaxis], npix, axis=2)  # Initialization
mag_matrix[:, :, 1, :] = np.repeat(A_initial[:, :, np.newaxis], npix, axis=2)

theta_ini, phi_ini = hp.pixelfunc.pix2ang(nside, range(npix))
beta_rays = np.zeros((2, npix, 3))
beta_rays[0, :, 0], beta_rays[1, :, 0] = theta_ini, phi_ini  # Initialization
beta_rays[0, :, 1], beta_rays[1, :, 1] = theta_ini, phi_ini

shear_matrix_neighbours = np.zeros((2, 2, 4, npix))

# the loop over all the shells (not the last as at each shell the contribution of the subsequent shells is calculated and we assume that after the final shell n(z) = 0.
for i_shell in tqdm(range(NR_shells - 1)):
    shear_matrix, deflec_field = shear_matrix_and_deflection_field_at_i(i_shell)
    if i_shell == 0:  # for the first plane, the rays are aimed exactly at the pixel centres
        ray_shear_matrix = np.array(
            ([shear_matrix[0], shear_matrix[1], shear_matrix[2]])
        )

    else:  # for the other planes we use bilinear interpolation or NGP to estimate the deflection field and shear matrix
        ray_coords = beta_rays[:, :, 1]
        ray_coords = ray_coords + 2 * np.pi
        ray_coords[1, :] = ray_coords[1, :] % (2 * np.pi)  # phi
        ray_coords[0, :] = ray_coords[0, :] % (
            np.pi
        )  # to make sure all in right range for healpix functions

        if not NGP:  # Bilinear interpolation
            neighbours, weights = hp.pixelfunc.get_interp_weights(
                nside, ray_coords[0, :], ray_coords[1, :], lonlat=False
            )
            shear_matrix_neighbours[0, 0, :, :] = shear_matrix[0][neighbours]
            shear_matrix_neighbours[1, 0, :, :] = shear_matrix[1][neighbours]
            shear_matrix_neighbours[0, 1, :, :] = shear_matrix[1][neighbours]
            shear_matrix_neighbours[1, 1, :, :] = shear_matrix[2][neighbours]
            transformed_shear_matrix_neighbours = transport_mag_matrices_neighbours(
                pixels_cart[:, neighbours], shear_matrix_neighbours
            )
            ray_shear_matrix[0, :] = np.sum(
                weights * transformed_shear_matrix_neighbours[0, 0, :, :], axis=0
            )
            ray_shear_matrix[1, :] = np.sum(
                weights
                * 0.5
                * (
                    transformed_shear_matrix_neighbours[0, 1, :, :]
                    + transformed_shear_matrix_neighbours[1, 0, :, :]
                ),
                axis=0,
            )
            ray_shear_matrix[2, :] = np.sum(
                weights * transformed_shear_matrix_neighbours[1, 1, :, :], axis=0
            )
            deflec_field = np.sum(weights * deflec_field[:, neighbours], axis=1)

        else:  # NGP
            closest = hp.pixelfunc.ang2pix(
                nside, ray_coords[0, :], ray_coords[1, :], nest=False, lonlat=False
            )
            shear_matrix = np.array(
                ([shear_matrix[0], shear_matrix[1]], [shear_matrix[1], shear_matrix[2]])
            )
            transformed_shear_matrix = transport_mag_matrix_NGP(
                pixels_cart[:, closest], shear_matrix
            )
            ray_shear_matrix[0, :] = transformed_shear_matrix[0, 0, :]
            ray_shear_matrix[1, :] = (
                transformed_shear_matrix[0, 1, :] + transformed_shear_matrix[1, 0, :]
            ) / 2
            ray_shear_matrix[2, :] = transformed_shear_matrix[1, 1, :]
            deflec_field = deflec_field[:, closest]

    factor = (
        plane_distances[i_shell]
        / plane_distances[i_shell + 1]
        * (plane_distances[i_shell + 1] - plane_distances[i_shell - 1])
        / (plane_distances[i_shell] - plane_distances[i_shell - 1])
    )  # For recurrance relations
    factor2 = (
        plane_distances[i_shell + 1] - plane_distances[i_shell]
    ) / plane_distances[i_shell + 1]

    # Compute magnification matrix on next plane for all the rays
    mag_matrix[0, 0, 2, :] = (
        (1 - factor) * mag_matrix[0, 0, 0, :]
        + factor * mag_matrix[0, 0, 1, :]
        - factor2
        * (
            ray_shear_matrix[0, :] * mag_matrix[0, 0, 1, :]
            + ray_shear_matrix[1, :] * mag_matrix[1, 0, 1, :]
        )
    )
    mag_matrix[1, 0, 2, :] = (
        (1 - factor) * mag_matrix[1, 0, 0, :]
        + factor * mag_matrix[1, 0, 1, :]
        - factor2
        * (
            ray_shear_matrix[1, :] * mag_matrix[0, 0, 1, :]
            + ray_shear_matrix[2, :] * mag_matrix[1, 0, 1, :]
        )
    )
    mag_matrix[0, 1, 2, :] = (
        (1 - factor) * mag_matrix[0, 1, 0, :]
        + factor * mag_matrix[0, 1, 1, :]
        - factor2
        * (
            ray_shear_matrix[0, :] * mag_matrix[0, 1, 1, :]
            + ray_shear_matrix[1, :] * mag_matrix[1, 1, 1, :]
        )
    )
    mag_matrix[1, 1, 2, :] = (
        (1 - factor) * mag_matrix[1, 1, 0, :]
        + factor * mag_matrix[1, 1, 1, :]
        - factor2
        * (
            ray_shear_matrix[1, :] * mag_matrix[0, 1, 1, :]
            + ray_shear_matrix[2, :] * mag_matrix[1, 1, 1, :]
        )
    )

    beta_rays[:, :, 2] = (
        (1 - factor) * beta_rays[:, :, 0]
        + factor * beta_rays[:, :, 1]
        - factor2 * deflec_field
    )

    if redshift_distribution:
        # When using a redshift distribution, add contribution to the final convergence map
        conv = (
            conv
            + (1.0 - 0.5 * (mag_matrix[0, 0, 2, :] + mag_matrix[1, 1, 2, :]))
            * n_chi[i_shell + 1]
            * delta_chis[i_shell + 1]
        )

    # Update the amplification matrices, angular coordinates
    mag_matrix[:, :, 0, :] = mag_matrix[:, :, 1, :]
    mag_matrix[:, :, 1, :] = mag_matrix[:, :, 2, :]

    beta_rays[:, :, 0] = beta_rays[:, :, 1]
    beta_rays[:, :, 1] = beta_rays[:, :, 2]


if not redshift_distribution:  # for n(z) = \delta(z=z of final shell)
    conv = 1.0 - 0.5 * (mag_matrix[0, 0, 2, :] + mag_matrix[1, 1, 2, :])

data_file_conv = h5py.File(f"./ff_test/{output_conv}", "w")
data_file_conv.create_dataset(
    "Convergence", data=conv, compression="gzip", chunks=True, shuffle=True
)
data_file_conv.close()


