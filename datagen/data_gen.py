import sapien
from data_utils import *
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Data generation for NeRF training.")

    parser.add_argument("--config", type=str, required=True, help="Path to configuration file.")
    parser.add_argument("--urdf_file", type=str, help="file path to the urdf file of sapien")
    parser.add_argument("--output_dir", type=str, help="path to save the generated images")
    parser.add_argument("--resolution", type=int, default=[512, 512], nargs='+', help="Image resolution, w h, default: w = 512, h = 512")
    parser.add_argument("--save_render_pose_path", type=str, default=None, help="path to save pose for rendering, default is None")
    parser.add_argument("--gen_art_imgs", type=bool, default=False, help="wheter to generate images with different articulation.")
    parser.add_argument("--art_nums", type=int, default=10, help="number of different articulation settings")
    
    parser.add_argument("--render_pose_path", type=str, default=None, help="load saved render pose for image generation, defalut is None")
    args = parser.parse_args()
    parser.add_argument("--qpos", type=float, nargs='+', default=None, help="set object articulation status, list of floats")
    parser.add_argument("--with_seg", type=bool, default=False, help="whether to save seg image with NeRF training data")
    parser.add_argument("--art_seg_data", type=bool, default=False, help="whether to set all qpos to their upper limit")
    parser.add_argument("--radius", type=int, default=4, help="camera distance to the origin")
    parser.add_argument("--q_pos", type=float, nargs="+", default=None, help="specify q_pos to the object")
    # Load and parse the JSON configuration file
    with open(args.config, "r") as config_file:
        config_data = json.load(config_file)

    required_args = ["urdf_file", "output_dir"]
    missing_args = [arg for arg in required_args if arg not in config_data]
    if missing_args:
        raise ValueError(f"Required argument(s) {', '.join(missing_args)} not found in the JSON configuration")

    # Update the args namespace with loaded JSON data
    for key, value in config_data.items():
        setattr(args, key, value)

    return args
        
def main(args):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer(offscreen_only=True)
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    
    urdf_path = args.urdf_file
    asset = loader.load_kinematic(str(urdf_path))
    # asset = loader.load(urdf_path)
    assert asset, 'URDF not loaded.'


    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    near, far = 0.1, 100
    width, height = args.resolution
    # width, height = 640, 480
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )

    output_path = P(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    qpos = getattr(args, 'q_pos', None)
    if args.gen_art_imgs:
        
        qpos_list = np.arange(10)
        scene_dict = {
                "scene": scene,
                "camera": camera,
                "asset": asset,
                "pose": None,
                "q_pos": None,
                "save_path": None,
                "save": None,
                "pose_fn": None,
                "camera_mount_actor": None
            }
        
        splits = ('train', 'val')
        generate_art_imgs(output_dir, 'train', qpos_list, scene_dict, 10, reuse_pose=True)
        generate_art_imgs(output_dir, 'val', qpos_list, scene_dict, 10)
    elif args.render_pose_path is not None:
        
        splits = ('train', 'test', 'val')
        for split in splits:
            generate_img_with_pose(args.render_pose_path, split, camera, asset, scene, object_path=output_path)
    elif args.art_seg_data:
        # set qpos to max
        gen_articulated_object_nerf_s2(30, args.radius, 'train', camera, asset, scene, object_path=output_path, \
        render_pose_file_dir=args.save_render_pose_path, q_pos=qpos)
        gen_articulated_object_nerf_s2(20, args.radius, 'val', camera, asset, scene, object_path=output_path, \
        render_pose_file_dir=args.save_render_pose_path, q_pos=qpos)
        generate_articulation_test(50, args.radius, 'test', camera, asset, scene, object_path=output_path, \
        render_pose_file_dir=args.save_render_pose_path)
    else:
        
        splits = ('train', 'test', 'val')
        print("generating images for training...")
        # gen_articulated_object_nerf_s1(120, 4, 'train', camera, asset, scene, object_path=output_path, render_pose_file_dir=args.save_render_pose_path, with_seg=args.with_seg)
        # print("generating images for validation...")
        # gen_articulated_object_nerf_s1(50, 4, 'test', camera, asset, scene, object_path=output_path, render_pose_file_dir=args.save_render_pose_path, with_seg=args.with_seg)
        # print("generating images for testing...")
        # gen_articulated_object_nerf_s1(50, 4, 'val', camera, asset, scene, object_path=output_path, render_pose_file_dir=args.save_render_pose_path, with_seg=args.with_seg)
        
        gen_articulated_object_nerf_s2(120, args.radius, 'train', camera, asset, scene, object_path=output_path, \
        render_pose_file_dir=args.save_render_pose_path, q_pos=qpos)
        gen_articulated_object_nerf_s2(50, args.radius, 'val', camera, asset, scene, object_path=output_path, \
        render_pose_file_dir=args.save_render_pose_path, q_pos=qpos)
        gen_articulated_object_nerf_s2(50, args.radius, 'test', camera, asset, scene, object_path=output_path, \
        render_pose_file_dir=args.save_render_pose_path, q_pos=qpos)

if __name__ == "__main__":
    args = parse_args()
    main(args)