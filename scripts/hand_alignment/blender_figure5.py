"""
blender_figure5.py  —  Reproduce the paper's Figure 5 ("Predicted hands placed
in the Gaussian scene") fully in Blender, headless or interactive.

Renders the reconstructed Gaussian field as an emissive colored point cloud
(unlit, like a splat) and drops the two predicted MANO hands in as lit meshes
(warm-pink left, glove-blue right), framed from the egocentric camera. The PLY
and the OBJs are parsed by hand so every vertex stays in the exact same
reconstruction world frame — no importer axis surprises.

Run headless:
  /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
    --python scripts/hand_alignment/blender_figure5.py -- \
    --dir fig_h2o_clip0 --frame 8 --out /tmp/fig5.png --res 1280 --samples 192

Open interactively (build scene, don't render, leave it for you to orbit):
  /Applications/Blender.app/Contents/MacOS/Blender --factory-startup \
    --python scripts/hand_alignment/blender_figure5.py -- --dir fig_h2o_clip0 --frame 8 --no-render
"""
import argparse
import json
import math
import os
import struct
import sys

import bpy  # noqa: E402
import numpy as np  # noqa: E402
import mathutils  # noqa: E402

SH_C0 = 0.28209479177387814


# ----------------------------- asset parsing ------------------------------ #
def parse_inria_ply(path):
    """Read the INRIA-format 3DGS PLY written by export_alignment_blender.py.
    Returns dict of arrays: xyz, rgb(0..1), alpha(0..1), radius(metres)."""
    with open(path, "rb") as fh:
        # header
        props = []
        n = 0
        while True:
            line = fh.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.frombuffer(fh.read(n * len(props) * 4), dtype="<f4").reshape(n, len(props))
    idx = {p: i for i, p in enumerate(props)}
    xyz = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float64)
    f_dc = data[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]]
    rgb = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)
    alpha = 1.0 / (1.0 + np.exp(-data[:, idx["opacity"]]))            # sigmoid
    scale = np.exp(data[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]])  # exp
    radius = scale.mean(axis=1)
    return {"xyz": xyz, "rgb": rgb.astype(np.float64),
            "alpha": alpha.astype(np.float64), "radius": radius.astype(np.float64)}


def parse_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("v "):
                verts.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    return np.asarray(verts, np.float64), np.asarray(faces, np.int32)


# ----------------------------- scene building ----------------------------- #
def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups):
        for blk in list(coll):
            coll.remove(blk)


def build_gaussian_cloud(g, opacity_thresh, radius_scale, radius_max, emission, lum_floor):
    keep = (g["alpha"] > opacity_thresh) & (g["rgb"].max(axis=1) > lum_floor)
    xyz, rgb, alpha = g["xyz"][keep], g["rgb"][keep], g["alpha"][keep]
    rad = np.clip(g["radius"][keep] * radius_scale, 1e-4, radius_max)
    print(f"[cloud] {keep.sum():,}/{keep.size:,} gaussians kept "
          f"(opacity>{opacity_thresh}, lum>{lum_floor})")

    mesh = bpy.data.meshes.new("gs_mesh")
    mesh.from_pydata([tuple(p) for p in xyz], [], [])
    mesh.update()
    col = mesh.color_attributes.new("col", "FLOAT_COLOR", "POINT")
    rgba = np.concatenate([rgb, alpha[:, None]], axis=1).astype(np.float32).ravel()
    col.data.foreach_set("color", rgba)
    ra = mesh.attributes.new("rad", "FLOAT", "POINT")
    ra.data.foreach_set("value", rad.astype(np.float32))

    obj = bpy.data.objects.new("gaussians", mesh)
    bpy.context.collection.objects.link(obj)

    # geometry nodes: mesh -> points (radius from "rad").
    # material: emission(col) alpha-blended with Transparent by per-point opacity, so
    # faint/edge gaussians fade instead of drawing as opaque speckles -> smooth splat look.
    mat = bpy.data.materials.new("gs_emit"); mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = "col"
    emit = nt.nodes.new("ShaderNodeEmission"); emit.inputs["Strength"].default_value = emission
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(attr.outputs["Alpha"], mix.inputs["Fac"])   # opacity drives blend
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])      # Fac=0 -> transparent
    nt.links.new(emit.outputs["Emission"], mix.inputs[2])    # Fac=1 -> emissive color
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    mod = obj.modifiers.new("m2p", "NODES")
    ng = bpy.data.node_groups.new("m2p_ng", "GeometryNodeTree")
    mod.node_group = ng
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput"); gout = ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    radattr = ng.nodes.new("GeometryNodeInputNamedAttribute")
    radattr.data_type = "FLOAT"; radattr.inputs["Name"].default_value = "rad"
    setmat = ng.nodes.new("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = mat
    ng.links.new(gin.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(radattr.outputs["Attribute"], m2p.inputs["Radius"])
    ng.links.new(m2p.outputs["Points"], setmat.inputs["Geometry"])
    ng.links.new(setmat.outputs["Geometry"], gout.inputs[0])
    return obj


def hand_material(name, base, subsurf, roughness=0.45):
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    for key, val in (("Subsurface Weight", subsurf),):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = val
    if "Subsurface Radius" in bsdf.inputs:
        bsdf.inputs["Subsurface Radius"].default_value = (base[0], base[1], base[2])
    return mat


def add_hand(path, name, mat):
    verts, faces = parse_obj(path)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(p) for p in verts], [], [tuple(f) for f in faces])
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(mat)
    bpy.context.collection.objects.link(obj)
    return obj


def setup_camera(extr_matrix, intr_matrix, res, dist, tilt_deg, lift, target, fov_mult,
                 ego, back, hero=False, extent=0.2, vfov=50.0, fill=0.55, zoom=1.0,
                 orbit_deg=0.0):
    C = np.asarray(extr_matrix, np.float64)          # cam2world (OpenCV)
    R, t = C[:3, :3], C[:3, 3]
    view_dir = R[:, 2]                                # OpenCV +Z (look direction)
    up_world = -R[:, 1]                               # Blender up
    target = np.asarray(target, np.float64)
    cam_data = bpy.data.cameras.new("cam")
    K = np.asarray(intr_matrix, np.float64)
    fx, cx = float(K[0, 0]), float(K[0, 2])
    cam_data.lens_unit = "FOV"
    cam_data.angle = 2.0 * math.atan(cx / fx) * fov_mult   # true FOV from principal point
    cam = bpy.data.objects.new("cam", cam_data)
    if hero:
        # Reproduce render_alignment_3d.py turntable angle-0 (his hero_000):
        # eye = target + unit_dir * radius, radius auto-fits the hand sphere to
        # `fill` of the vfov; orientation = ego roll, hand centred (look-at target).
        cam_data.sensor_fit = "VERTICAL"
        cam_data.angle = math.radians(vfov)
        ud = (t - target); ud = ud / (np.linalg.norm(ud) + 1e-9)
        radius = extent / (math.tan(math.radians(vfov) / 2.0) * fill) * zoom
        eye0 = target + ud * radius
        # Keep the EGO orientation (so the background is the headset's forward view)
        # and centre the hand with a principal-point shift — exactly the turntable's
        # angle-0 (R = R0, fixed (cx,cy) crop), not a look-at.
        Rb = R @ np.diag([1.0, -1.0, -1.0])
        M = np.eye(4); M[:3, :3] = Rb; M[:3, 3] = eye0
        cam.matrix_world = mathutils.Matrix(M.tolist())
        tc0 = R.T @ (target - eye0)                 # target in OpenCV cam frame
        tx, ty, tz = float(tc0[0]), float(tc0[1]), float(tc0[2])
        half_t = math.tan(math.radians(vfov) / 2.0)
        cam_data.shift_x = -tx / (tz * half_t) / 2.0
        cam_data.shift_y = ty / (tz * half_t) / 2.0
    elif ego:
        # headset's own position (pulled back a touch), aimed at the hands so they
        # stay centred — reproduces the first-person "looking at my hands" framing.
        pos = t - back * view_dir + lift * up_world
        fwd = mathutils.Vector(tuple(target - pos))
        M = fwd.to_track_quat("-Z", "Y").to_matrix().to_4x4()
        M.translation = mathutils.Vector(tuple(pos))
        cam.matrix_world = M
    else:
        # external look-at toward the hand midpoint; --orbit swings the viewpoint
        # around the camera-up axis (0 = straight along the ego view direction)
        vd = view_dir
        if orbit_deg:
            q = mathutils.Quaternion(mathutils.Vector(tuple(up_world)),
                                     math.radians(orbit_deg))
            vd = np.asarray(q.to_matrix()) @ view_dir
        pos = target - dist * vd + lift * up_world
        fwd = mathutils.Vector(tuple(target - pos))
        M = fwd.to_track_quat("-Z", "Y").to_matrix().to_4x4()
        M.translation = mathutils.Vector(tuple(pos))
        cam.matrix_world = M
    if tilt_deg:
        cam.matrix_world = cam.matrix_world @ mathutils.Matrix.Rotation(
            math.radians(tilt_deg), 4, "X")
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def add_lights(center, up, light_scale=1.0):
    up = np.asarray(up, np.float64)
    key = bpy.data.lights.new("key", "AREA"); key.energy = 220 * light_scale; key.size = 1.6
    ko = bpy.data.objects.new("key", key)
    ko.location = tuple(np.asarray(center) + up * 1.4 + np.array([0.8, 0.0, -0.6]))
    bpy.context.collection.objects.link(ko)
    fill = bpy.data.lights.new("fill", "AREA"); fill.energy = 90 * light_scale; fill.size = 2.2
    fo = bpy.data.objects.new("fill", fill)
    fo.location = tuple(np.asarray(center) + up * 0.6 + np.array([-1.1, 0.2, -0.4]))
    bpy.context.collection.objects.link(fo)
    for o in (ko, fo):       # aim at center
        d = mathutils.Vector(tuple(np.asarray(center) - np.asarray(o.location)))
        o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--frame", type=int, default=8)
    ap.add_argument("--out", default="/tmp/fig5.png")
    ap.add_argument("--res", type=int, default=1280)
    ap.add_argument("--samples", type=int, default=192)
    ap.add_argument("--opacity_thresh", type=float, default=0.10)
    ap.add_argument("--radius_scale", type=float, default=2.2)
    ap.add_argument("--radius_max", type=float, default=0.012)
    ap.add_argument("--lum_floor", type=float, default=0.0, help="drop near-black gaussians")
    ap.add_argument("--emission", type=float, default=1.05)
    ap.add_argument("--dist", type=float, default=0.55, help="camera distance from hand midpoint")
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--lift", type=float, default=0.12)
    ap.add_argument("--fov_mult", type=float, default=1.0)
    ap.add_argument("--ego", action="store_true", help="render from the headset's own pose")
    ap.add_argument("--back", type=float, default=0.0, help="ego pull-back along view dir")
    ap.add_argument("--hero", action="store_true", help="reproduce render_alignment_3d hero_000")
    ap.add_argument("--vfov", type=float, default=50.0)
    ap.add_argument("--fill", type=float, default=0.55)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--light_scale", type=float, default=1.0,
                    help="scale key/fill light energy (small scenes blow out at 1.0)")
    ap.add_argument("--bg", type=float, default=0.05, help="world background grey value")
    ap.add_argument("--bg_strength", type=float, default=0.35)
    ap.add_argument("--film_white", action="store_true",
                    help="transparent film composited over white (white holes, "
                         "lighting unchanged)")
    ap.add_argument("--orbit", type=float, default=0.0,
                    help="swing the look-at camera around the up axis (deg)")
    ap.add_argument("--matte", action="store_true",
                    help="clay-look hands (no subsurface, saturated palette)")
    ap.add_argument("--hand_clear", type=float, default=0.0,
                    help="drop gaussians within this distance of any hand vertex "
                         "(scene units) so the meshes aren't speckled")
    ap.add_argument("--save_blend", default=None, help="also write a .blend you can open")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)

    d = args.dir
    be = os.path.join(d, "blender_export")
    ply = os.path.join(be, f"gaussians_frame{args.frame:04d}.ply")
    hl = os.path.join(be, f"hand_frame{args.frame:04d}_left.obj")
    hr = os.path.join(be, f"hand_frame{args.frame:04d}_right.obj")
    cam_json = json.load(open(os.path.join(d, "camera_params.json")))
    extr = cam_json["extrinsics"][args.frame]["matrix"]
    intr = cam_json["intrinsics"][args.frame]["matrix"]

    clear_scene()
    g = parse_inria_ply(ply)
    if args.hand_clear > 0:
        hv_all = [parse_obj(p)[0] for p in (hl, hr) if os.path.exists(p)]
        if hv_all:
            hv_all = np.concatenate(hv_all, 0)
            keep = np.ones(len(g["xyz"]), bool)
            for i in range(0, len(g["xyz"]), 8192):   # chunked pairwise distances
                d = np.linalg.norm(g["xyz"][i:i + 8192, None, :] - hv_all[None], axis=2)
                keep[i:i + 8192] = d.min(axis=1) > args.hand_clear
            print(f"[hand_clear] dropped {(~keep).sum():,} gaussians near hands")
            g = {k: v[keep] for k, v in g.items()}
    build_gaussian_cloud(g, args.opacity_thresh, args.radius_scale, args.radius_max,
                         args.emission, args.lum_floor)

    if args.matte:
        skin_mat = hand_material("skin", (0.87, 0.50, 0.44), 0.0, roughness=0.65)
        glove_mat = hand_material("glove", (0.45, 0.66, 0.88), 0.0, roughness=0.65)
    else:
        skin_mat = hand_material("skin", (0.83, 0.46, 0.40), 0.25)
        glove_mat = hand_material("glove", (0.18, 0.46, 0.80), 0.0)
    hand_pts = []
    if os.path.exists(hl):
        add_hand(hl, "hand_left", skin_mat)
        hand_pts.append(parse_obj(hl)[0])
    if os.path.exists(hr):
        add_hand(hr, "hand_right", glove_mat)
        hand_pts.append(parse_obj(hr)[0])
    if hand_pts:
        hv = np.concatenate(hand_pts, 0)
        target = hv.mean(0)
        extent = float(np.linalg.norm(hv - target, axis=1).max())
    else:
        target = g["xyz"].mean(0); extent = 0.2

    up_world = -np.asarray(extr, np.float64)[:3, 1]
    setup_camera(extr, intr, args.res, args.dist, args.tilt, args.lift, target,
                 args.fov_mult, args.ego, args.back,
                 hero=args.hero, extent=extent, vfov=args.vfov, fill=args.fill,
                 zoom=args.zoom, orbit_deg=args.orbit)
    add_lights(target, up_world, args.light_scale)

    world = bpy.data.worlds.new("w"); world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (args.bg, args.bg, args.bg, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = args.bg_strength
    bpy.context.scene.world = world

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    try:
        sc.cycles.device = "GPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = True
    except Exception as e:
        print("[cycles] GPU setup skipped:", e)
    sc.cycles.samples = args.samples
    sc.cycles.transparent_max_bounces = 128   # let many overlapping splats blend
    sc.cycles.max_bounces = 8
    sc.render.resolution_x = args.res
    sc.render.resolution_y = args.res
    sc.render.film_transparent = args.film_white
    if args.film_white:
        sc.render.image_settings.color_mode = "RGBA"   # composite over white afterwards
    sc.view_settings.view_transform = "AgX" if "AgX" in [v.name for v in bpy.types.ColorManagedViewSettings.bl_rna.properties['view_transform'].enum_items] else "Standard"

    # open every 3D viewport in Rendered shading, looking THROUGH the camera, so the
    # saved .blend shows the exact framed figure on load (no navigating needed).
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "RENDERED"
                        space.region_3d.view_perspective = "CAMERA"

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend))
        print("WROTE_BLEND", os.path.abspath(args.save_blend))

    if not args.no_render:
        sc.render.filepath = args.out
        bpy.ops.render.render(write_still=True)
        print("WROTE", args.out)


if __name__ == "__main__":
    main()
