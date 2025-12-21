# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains import FlatPatchSamplingCfg

ROUGH_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.1, 0.5), platform_width=2.0, border_width=0.25,
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
        #             num_patches=100, patch_radius=0.05, max_height_diff=0.1)
        #     },
        # ),
        # "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.1, 0.5), platform_width=2.0, border_width=0.25,
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
        #             num_patches=100, patch_radius=0.05, max_height_diff=0.1)
        #     },
        # ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.05, max_height_diff=0.1)
            },
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.05, max_height_diff=0.1)
            },
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.1, max_height_diff=0.1)
            },
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.1, max_height_diff=0.1)
            },
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.15), platform_width=3.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.05, max_height_diff=0.1)
            },
        ),
    }, 
)

GAP_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.5,
            gap_width_range=(0.1, 1.2),
            platform_width=3.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.25, max_height_diff=0.1)
            },
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.05, max_height_diff=0.1)
            },
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.1, max_height_diff=0.1)
            },
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.1, max_height_diff=0.1)
            },
        ),
    }, 
)

PIT_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.5,
            pit_depth_range=(0.05, 0.9),
            platform_width=3.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.25, max_height_diff=0.1)
            },
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.05, max_height_diff=0.1)
            },
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.1, max_height_diff=0.1)
            },
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    x_range=(-4.5, 4.5), y_range=(-4.5, 4.5),
                    num_patches=100, patch_radius=0.1, max_height_diff=0.1)
            },
        ),
    },
)