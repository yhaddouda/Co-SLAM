#!/usr/bin/env python3
"""Common post-SLAM reconstruction scorer; never used for neural timing gains.

Retains GT-pose/virtual-view culling, point ICP, 200k surface samples, 5cm
completion and the supplied 1000 metric views. Corrects geometry isolation,
missing-depth handling and cache provenance. Original inputs remain read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('PYOPENGL_PLATFORM','egl')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


class Rasterizer:
    def __init__(self,width,height,K,near=.1,far=20.):
        import numpy as np
        import pyrender
        self.np=np;self.pyrender=pyrender;self.mesh_node=None
        self.scene=pyrender.Scene(bg_color=np.zeros(4))
        self.camera=self.scene.add(pyrender.IntrinsicsCamera(fx=K[0,0],fy=K[1,1],cx=K[0,2],cy=K[1,2],znear=near,zfar=far),pose=np.eye(4))
        self.renderer=pyrender.OffscreenRenderer(width,height)
        self.width=width;self.height=height

    def set_mesh(self,mesh):
        if self.mesh_node is not None:self.scene.remove_node(self.mesh_node)
        self.mesh_node=None
        if mesh is not None:
            self.mesh_node=self.scene.add(self.pyrender.Mesh.from_trimesh(mesh,smooth=False))
        assert len(self.scene.mesh_nodes)==(0 if mesh is None else 1)

    def render(self,pose,opencv=False):
        pose=self.np.array(pose,copy=True)
        if opencv:pose[:3,1:3]*=-1
        self.scene.set_pose(self.camera,pose)
        # pyrender 0.1.45's DEPTH_ONLY branch unconditionally culls back faces,
        # ignoring SKIP_CULL_FACES. The normal path honors it; discard color.
        flags=self.pyrender.RenderFlags.SKIP_CULL_FACES
        _,depth=self.renderer.render(self.scene,flags=flags)
        depth=self.np.asarray(depth,dtype=self.np.float32)
        depth[~self.np.isfinite(depth)]=0
        return depth

    def close(self):
        self.renderer.delete()


def project(points,pose,K,width,height,depth,eps=.03):
    """The legacy cull_from_one_pose equations, using OpenGL c2w input."""
    import numpy as np
    cv=np.array(pose,copy=True);cv[:3,1:3]*=-1
    w2c=np.linalg.inv(cv)
    camera=w2c[:3,:3]@points.T+w2c[:3,3,None]
    uvz=(K@camera).T
    z=uvz[:,2]+1e-8;x=uvz[:,0]/z;y=uvz[:,1]/z
    inside=(x>=0)&(x<=width-1)&(y>=0)&(y<=height-1)&(z>0)
    u=np.clip(x,0,width-1).astype(np.int32);v=np.clip(y,0,height-1).astype(np.int32)
    visible=inside&(z<depth[v,u]+eps)
    return inside,visible


def cull(mesh,asset,output):
    import numpy as np
    import trimesh
    with np.load(asset['cull_poses']) as data:
        poses=data['poses'];real_count=int(data['real_count'])
    K=np.asarray(asset['K'],float);height,width=asset['height'],asset['width']
    raster=Rasterizer(width,height,K,near=.01,far=10.)
    in_frustum=np.zeros(len(mesh.vertices),bool);observed=np.zeros(len(mesh.vertices),bool)
    raster.set_mesh(mesh)
    try:
        for i,pose in enumerate(poses):
            depth=raster.render(pose)
            inside,visible=project(mesh.vertices,pose,K,width,height,depth)
            if i<real_count:in_frustum|=inside
            observed|=visible
            if i%100==0:print('CULL',i,len(poses),flush=True)
    finally:raster.close()
    keep_vertex=in_frustum&observed
    # Native rule: any vertex in the real frusta AND any observed vertex.
    # These need not be the same vertex, so combine at the triangle level.
    faces=mesh.faces
    keep=in_frustum[faces].any(1)&observed[faces].any(1)
    result=trimesh.Trimesh(mesh.vertices,faces[keep],process=False)
    result.remove_unreferenced_vertices()
    if not len(result.faces):raise RuntimeError('Culling produced an empty mesh')
    result.export(output)
    return result,dict(input_faces=len(faces),culled_faces=len(result.faces),real_views=real_count,virtual_views=len(poses)-real_count,epsilon_m=.03)


def gt_cache(asset,cache):
    import numpy as np
    import trimesh
    import pyrender
    protocol=dict(version='isolated_double_sided_egl_v2',gt_sha256=sha(asset['gt_mesh']),poses_sha256=sha(asset['metric_poses']),scorer_sha256=sha(__file__),pyrender_version=pyrender.__version__,height=500,width=500,fx=300.,fy=300.,cx=249.5,cy=249.5,near=.1,far=20.,pose_convention='OpenCV; converted to OpenGL for EGL',missing_depth=0.,geometry='exactly one mesh')
    key=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    directory=Path(cache)/key;directory.mkdir(parents=True,exist_ok=True)
    data=directory/'depths.npy';metadata=directory/'complete.json'
    if metadata.exists():
        info=json.loads(metadata.read_text())
        if info['protocol']!=protocol or info['depths_sha256']!=sha(data):raise RuntimeError('GT cache provenance mismatch')
        depths=np.load(data,mmap_mode='r')
        assert depths.shape==(1000,500,500) and depths.dtype==np.float32
        return depths,key
    if data.exists():raise RuntimeError('Incomplete GT cache retained: '+str(directory))
    poses=np.load(asset['metric_poses'])['poses'];assert poses.shape==(1000,4,4)
    depths=np.lib.format.open_memmap(data,mode='w+',dtype=np.float32,shape=(1000,500,500))
    raster=Rasterizer(500,500,np.array([[300.,0,249.5],[0,300.,249.5],[0,0,1.]]))
    raster.set_mesh(trimesh.load(asset['gt_mesh'],force='mesh',process=False))
    try:
        for i,pose in enumerate(poses):
            depths[i]=raster.render(pose,opencv=True)
            if i%100==0:print('GT_DEPTH',i,1000,flush=True)
    finally:raster.close()
    depths.flush()
    save_json(metadata,dict(protocol=protocol,depths_sha256=sha(data)))
    return np.load(data,mmap_mode='r'),key


def geometry_metrics(rec,gt):
    import numpy as np
    import open3d as o3d
    import trimesh
    from scipy.spatial import cKDTree
    a=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rec.vertices))
    b=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt.vertices))
    transform=o3d.pipelines.registration.registration_icp(a,b,.1,np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPoint()).transformation
    aligned=rec.copy();aligned.apply_transform(transform)
    np.random.seed(0)
    rec_points=trimesh.sample.sample_surface(aligned,200000)[0]
    gt_points=trimesh.sample.sample_surface(gt,200000)[0]
    accuracy=cKDTree(gt_points).query(rec_points)[0]
    completion=cKDTree(rec_points).query(gt_points)[0]
    metrics=dict(accuracy_cm=float(100*accuracy.mean()),completion_cm=float(100*completion.mean()),completion_ratio_pct=float(100*(completion<.05).mean()))
    return aligned,transform,metrics


def evaluate(job,output,cache):
    import numpy as np
    import trimesh
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    asset=json.loads(Path(job['asset']).read_text())
    for key,value in asset['hashes'].items():
        if sha(key)!=value:raise RuntimeError('Evaluation asset changed: '+key)
    mesh_path=Path(job['mesh']);input_hash=sha(mesh_path)
    mesh=trimesh.load(mesh_path,force='mesh',process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():raise RuntimeError('Input mesh empty/nonfinite')
    begin=time.time()
    culled,cull_meta=cull(mesh,asset,output/'culled_mesh.ply')
    gt=trimesh.load(asset['gt_mesh'],force='mesh',process=False)
    aligned,transform,metrics=geometry_metrics(culled,gt)
    gt_depths,cache_key=gt_cache(asset,cache)
    poses=np.load(asset['metric_poses'])['poses']
    raster=Rasterizer(500,500,np.array([[300.,0,249.5],[0,300.,249.5],[0,0,1.]]))
    raster.set_mesh(aligned)
    errors=[];valid_views=[];valid_pixels=[]
    try:
        for i,pose in enumerate(poses):
            ours=raster.render(pose,opencv=True)
            target=gt_depths[i]
            mask=(target>0)&(target<19.)
            n=int(mask.sum());valid_pixels.append(n)
            if n>100:
                errors.append(float(np.abs(target[mask]-ours[mask]).mean()))
                valid_views.append(i)
            if i%100==0:print('REC_DEPTH',i,1000,flush=True)
    finally:raster.close()
    if not errors:raise RuntimeError('No valid depth views')
    metrics['depth_l1_cm']=float(np.mean(errors)*100)
    assert all(np.isfinite(v) for v in metrics.values())
    result=dict(**{k:job[k] for k in ('id','system','dataset','scene','condition','hash_size','seed')},metrics=metrics,input_mesh=str(mesh_path),input_mesh_sha256=input_hash,asset_sha256=sha(job['asset']),source_manifest_sha256=job['source_manifest_sha256'],scorer_sha256=sha(__file__),culling=cull_meta,icp_transform=transform.tolist(),gt_cache_key=cache_key,depth_valid_views=len(errors),depth_valid_pixel_counts=valid_pixels,wall_seconds=time.time()-begin,protocol='checked_common_egl_v2',post_slam_only=True)
    save_json(output/'metrics.json',result)
    np.savez(output/'depth_errors.npz',valid_view_ids=valid_views,absolute_depth_error_m=errors)
    print('RECON_COMPLETE',job['id'],json.dumps(metrics),flush=True)


def validate(output):
    import numpy as np
    import trimesh
    from OpenGL import GL
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    K=np.array([[40.,0,31.5],[0,40.,31.5],[0,0,1.]])
    def plane(z,reverse=False):
        faces=np.array([[0,1,2],[0,2,3]])
        if reverse:faces=faces[:,::-1]
        return trimesh.Trimesh(np.array([[-2,-2,z],[2,-2,z],[2,2,z],[-2,2,z]]),faces,process=False)
    raster=Rasterizer(64,64,K)
    try:
        raster.set_mesh(plane(2));a=raster.render(np.eye(4),opencv=True)
        raster.renderer._platform.make_current()
        hardware={k:GL.glGetString(v).decode() for k,v in [('renderer',GL.GL_RENDERER),('vendor',GL.GL_VENDOR),('version',GL.GL_VERSION)]}
        raster.renderer._platform.make_uncurrent()
        raster.set_mesh(plane(2,True));b=raster.render(np.eye(4),opencv=True)
        raster.set_mesh(plane(4));far=raster.render(np.eye(4),opencv=True)
        raster.set_mesh(None);empty=raster.render(np.eye(4),opencv=True)
        print('PLANE_CALIBRATION',dict(near=float(a[32,32]),reverse=float(b[32,32]),far=float(far[32,32]),empty_max=float(empty.max())),flush=True)
        np.savez(output/'plane_debug.npz',near=a,reverse=b,far=far,empty=empty)
        assert abs(a[32,32]-2)<1e-3 and abs(far[32,32]-4)<1e-3
        assert np.max(abs(a-b))<1e-5 and not empty.any()
        assert abs(far[32,32]-a[32,32]-2)<1e-3
        assert 'NVIDIA' in hardware['vendor'] and '4070' in hardware['renderer'],hardware
        # GT==reconstruction must give zero with the exact same valid mask.
        assert float(np.abs(a[a>0]-b[a>0]).mean())<1e-6
        pose=np.diag([1.,-1.,-1.,1.])
        points=np.array([[0,0,2],[0,0,3],[100,0,2], [0,0,-1]])
        inside,observed=project(points,pose,K,64,64,a)
        assert inside.tolist()==[True,True,False,False]
        assert observed.tolist()==[True,False,False,False]
    finally:raster.close()
    save_json(output/'validation_passed.json',dict(status='PASSED',plane_depth_calibration=True,double_sided=True,geometry_isolation=True,empty_scene_missing_depth_zero=True,gt_identity_depth_zero=True,culling_projection_and_occlusion=True,hardware=hardware,scorer_sha256=sha(__file__)))
    print('RECON_VALIDATION_PASSED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    if args.validate:validate(args.output)
    else:evaluate(json.loads(args.job.read_text()),args.output,args.cache)


if __name__=='__main__':main()
