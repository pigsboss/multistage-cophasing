# -*- coding: utf-8 -*-
"""
Debug tool for the **visualizers** module itself.
Prints the scene‑graph hierarchy and, for solar‑system or earth‑moon scenes,
creates a 2D scatter plot of celestial body **display positions** using matplotlib.
No 3D rendering or interactive GUI window (matplotlib window is optional).
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple
import sys
from pathlib import Path
from ..base import (
    Scene, SceneNode, Group, ScaledGroup,
    Ellipsoid, Arrow, Trajectory, Camera, Background,
)
from . import Renderer


class DebugRenderer(Renderer):
    """
    Pure debug backend:
    1. Always prints the scene tree.
    2. For scenes with celestial bodies (solar‑system, sun‑earth‑moon, etc.),
       shows a top‑down matplotlib scatter plot of the (display‑space) positions.
    Does NOT require vedo and does NOT open any 3D window.
    """

    def render(self, scene: Scene, **kwargs) -> None:
        print(f"DEBUG RENDERING: scene = '{scene.name}'  time = {scene.time:.3f} s")
        self._print_node(scene.root, indent=0)

        # Always attempt to plot; the method will exit gracefully if no ellipsoids found.
        self._plot_ellipsoid_positions(scene, **kwargs)

    # ------------------------------------------------------------------
    # Tree printing (unchanged)
    # ------------------------------------------------------------------
    def _print_node(self, node: SceneNode, indent: int = 0):
        prefix = "  " * indent
        type_name = type(node).__name__
        print(f"{prefix}{type_name}: '{node.name}'")
        if isinstance(node, Ellipsoid):
            print(f"{prefix}  radii = {node.radii}")
        elif isinstance(node, Arrow):
            print(f"{prefix}  dir = {node.direction}, "
                  f"length = {node.length}, color = {node.color}")
        elif isinstance(node, Trajectory):
            print(f"{prefix}  points count = {len(node.points)}")
        elif isinstance(node, ScaledGroup):
            print(f"{prefix}  scale function = "
                  f"{node.scale_function.__class__.__name__}")
        elif isinstance(node, Camera):
            print(f"{prefix}  target = {node.target}, fov = {node.fov}")
        for child in node.children:
            self._print_node(child, indent + 1)

    # ------------------------------------------------------------------
    # Matplotlib scatter plot (supports both interactive and file‑mode)
    # ------------------------------------------------------------------
    def _plot_ellipsoid_positions(self, scene: Scene, **kwargs):
        """Recursively collect all Ellipsoid nodes from the scene graph and plot
        their world‑space (display) positions as a 2D scatter plot.
        If 'output_dir' is given in kwargs, save to PNG instead of showing.
        The unit (AU or LD) is chosen automatically based on the scene name.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        output_dir = kwargs.get("output_dir", None)
        frame_index = kwargs.get("frame_index", 0)

        # ------ Collect bodies (world positions) ------
        bodies: List[Tuple[str, np.ndarray, str]] = []
        self._collect_ellipsoid_nodes(scene.root, np.zeros(3), bodies)

        if not bodies:
            print("[DEBUG] No Ellipsoid nodes found in the scene – nothing to plot.")
            return

        names = [b[0] for b in bodies]
        cols = [b[2] for b in bodies]

        # Determine unit and appropriate axis limits based on scene
        scene_lower = scene.name.lower()
        if "solar" in scene_lower:
            unit_label = "AU"
            unit_factor = 149597870700.0   # 1 AU in metres
            xlim = ylim = (-15, 15)
        else:
            # Default: treat as Earth‑Moon or any small‑scale scene
            unit_label = "LD"
            unit_factor = 3.844e8          # 1 Lunar Distance in metres
            xlim = ylim = (-5, 5)

        xs = [b[1][0] / unit_factor for b in bodies]
        ys = [b[1][1] / unit_factor for b in bodies]

        # If saving files, use a non‑interactive backend to avoid window pop‑ups
        if output_dir is not None:
            matplotlib.use("Agg")
            plt.ioff()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Scene: {scene.name} (Scaled Positions)", fontsize=14)
        ax.set_xlabel(f"X [{unit_label}]")
        ax.set_ylabel(f"Y [{unit_label}]")
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # Plot Sun (or any central body) larger, others smaller
        for name, x, y, col in zip(names, xs, ys, cols):
            if name.lower() == 'sun':
                ax.scatter(x, y, c=col, s=200, edgecolors='white',
                           linewidth=0.5, label=name, zorder=10)
            else:
                ax.scatter(x, y, c=col, s=80, edgecolors='black',
                           linewidth=0.5, label=name, zorder=5)

        # Labels slightly offset
        for name, x, y in zip(names, xs, ys):
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, alpha=0.9)

        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        fig.tight_layout()

        if output_dir is not None:
            filename = Path(output_dir) / f"frame_{frame_index:04d}.png"
            fig.savefig(str(filename), dpi=150)
            plt.close(fig)
            print(f"[DEBUG] Saved {filename}", file=sys.stderr)
        else:
            plt.show()

    @staticmethod
    def _collect_ellipsoid_nodes(node: SceneNode, parent_pos: np.ndarray,
                                 out: List[Tuple[str, np.ndarray, str]]):
        """Depth‑first traversal that accumulates world positions.
        When an Ellipsoid is encountered, its world position is recorded.
        """
        world_pos = parent_pos + node.transform.position

        if isinstance(node, Ellipsoid):
            colour = getattr(node, 'color', 'white')
            out.append((node.name, world_pos, colour))
            # Ellipsoids usually have no children, but we continue just in case.
        for child in node.children:
            DebugRenderer._collect_ellipsoid_nodes(child, world_pos, out)
