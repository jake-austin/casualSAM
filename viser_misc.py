import viser
import numpy as np
import viser.transforms as tf
from tqdm import tqdm
from typing import List
import time
import skimage
from typing import Tuple, cast


def get_point_cloud(rgb, R, T, K, depth, mask = None, downsample_factor: int = 1):
    rgb = rgb[::downsample_factor, ::downsample_factor]
    depth = skimage.transform.resize(depth, rgb.shape[:2], order=0)
    if mask is not None:
        mask = cast(
            np.typing.NDArray[np.typing.bool_],
            skimage.transform.resize(mask, rgb.shape[:2], order=0),
        )
        assert depth.shape == rgb.shape[:2]
    else:
        mask = np.ones(rgb.shape[:2], dtype=np.bool_)

    # T_world_camera = tf.SE3.from_rotation_and_translation(
    #     tf.SO3.from_matrix(R), T
    # ).as_matrix()
        
    T_world_camera = np.eye(4)
    T_world_camera[:3, :3] = R
    T_world_camera[:3, 3] = T

    img_wh = rgb.shape[:2][::-1]

    grid = np.stack(np.meshgrid(range(img_wh[0]), range(img_wh[1])), 2) + 0.5
    grid = grid * downsample_factor

    homo_grid = np.pad(grid[mask], ((0, 0), (0, 1)), constant_values=1)
    local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
    dirs = np.einsum("ij,bj->bi", T_world_camera[:3, :3], local_dirs)
    points = (T_world_camera[:3, 3] + dirs * depth[mask, None]).astype(np.float32)
    point_colors = rgb[mask]

    return points, point_colors


class TemporalServer(viser.ViserServer):

    def setup(self, R, T, K, imgs, depths, uncertainty=None):

        num_frames = R.shape[0]-1

        # Add playback UI.
        with self.add_gui_folder("Playback"):
            gui_timestep = self.add_gui_slider(
                "Timestep",
                min=0,
                max=num_frames,
                step=1,
                initial_value=0,
                disabled=True,
            )

            gui_next_frame = self.add_gui_button("Next Frame", disabled=True)
            gui_prev_frame = self.add_gui_button("Prev Frame", disabled=True)
            gui_playing = self.add_gui_checkbox("Playing", True)
            gui_accumulate = self.add_gui_checkbox("Accumulate Pointclouds", False)
            gui_framerate = self.add_gui_slider(
                "FPS", min=1, max=60, step=0.1, initial_value=10
            )
            gui_framerate_options = self.add_gui_button_group(
                "FPS options", ("10", "20", "30", "60")
            )

        gui_reset_up = self.add_gui_button(
            "Reset up direction",
            hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )
        @gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                [0.0, -1.0, 0.0]
            )


        # Frame step buttons.
        @gui_next_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        @gui_prev_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value - 1) % num_frames

        # Disable frame controls when we're playing.
        @gui_playing.on_update
        def _(_) -> None:
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        # Set the framerate when we click one of the options.
        @gui_framerate_options.on_click
        def _(_) -> None:
            gui_framerate.value = int(gui_framerate_options.value)

        prev_timestep = gui_timestep.value

        @gui_accumulate.on_update
        def _(_) -> None:
            # Make all frames visible or invisible
            if gui_accumulate.value:
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = True
            else:
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = i == gui_timestep.value
            

        # Toggle frame visibility when the timestep slider changes.
        @gui_timestep.on_update
        def _(_) -> None:
            nonlocal prev_timestep
            current_timestep = gui_timestep.value
            with self.atomic():
                frame_nodes[current_timestep].visible = True
                frame_nodes[prev_timestep].visible = gui_accumulate.value
            prev_timestep = current_timestep


        # Load in frames.
        self.add_frame(
            "/frames",
            wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
            position=(0, 0, 0),
            show_axes=False,
        )

        frame_nodes: List[viser.FrameHandle] = []
        for i in tqdm(range(num_frames)):
            position, color = get_point_cloud(imgs[i], R[i], T[i], K[i], depths[i])

            # Add base frame.
            frame_nodes.append(self.add_frame(f"/frames/t{i}", show_axes=False))

            # Place the point cloud in the frame.
            self.add_point_cloud(
                name=f"/frames/t{i}/point_cloud",
                points=position,
                colors=color,
                point_size=0.001,
            )

            # Place the frustum.
            fov = 2 * np.arctan2(imgs[i].shape[0] / 2, K[i, 0, 0])
            aspect = imgs[i].shape[1] / imgs[i].shape[0]
            self.add_camera_frustum(
                f"/frames/t{i}/frustum",
                fov=fov,
                aspect=aspect,
                scale=0.015,
                image=imgs[i, ::4, ::4],
                wxyz=tf.SO3.from_matrix(R[i]).wxyz,
                position=T[i],
            )

            # Add some axes.
            self.add_frame(
                f"/frames/t{i}/frustum/axes",
                axes_length=0.005,
                axes_radius=0.0005,
            )


        # Hide all but the current frame.
        for i, frame_node in enumerate(frame_nodes):
            frame_node.visible = i == gui_timestep.value

        # Playback update loop.
        prev_timestep = gui_timestep.value
        while True:
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames

            time.sleep(1.0 / gui_framerate.value)



"""


import numpy as np
import glob
from viser_misc import TemporalServer

#not fine
causalSAM_fps = sorted(glob.glob('/home/jake-austin/casualSAM/stride_2/prosthetic-swimmer/03-01/local_dev/prosthetic-swimmer/BA_full/*.npz'))
#not fine
causalSAM_fps = sorted(glob.glob('/home/jake-austin/casualSAM/no_depth_finetune_v3/prosthetic-swimmer/03-01/local_dev/prosthetic-swimmer/BA_full/*.npz'))
#fine, clearly this is an issue with the low memory bundle adjustment function
causalSAM_fps = sorted(glob.glob('/home/jake-austin/casualSAM/no_depth_finetune_v3/prosthetic-swimmer/03-01/local_dev/prosthetic-swimmer/init_window_BA/*.npz'))


imgs = np.stack([np.load(p)['img'] for p in causalSAM_fps], axis=0)
R = np.stack([np.load(p)['R'] for p in causalSAM_fps], axis=0)
T = np.stack([np.load(p)['t'] for p in causalSAM_fps], axis=0)
K = np.stack([np.load(p)['K'] for p in causalSAM_fps], axis=0)
depths = np.stack([1 / (np.load(p)['disp']+1e-6) for p in causalSAM_fps], axis=0)

stride = 4
K[:, :2, :] /= stride

server = TemporalServer()
server.setup(R, T, K, imgs[:, ::stride, ::stride], depths[:, ::stride, ::stride])



"""
