"""Build an evaluator-only bundle from frozen box hypotheses.

This adapter never feeds ground truth into reconstruction. It joins the private
scene file only after predictions are frozen, and marks axis matching as oracle
when the hypothesis has no independently inferred up direction.
"""
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); gt_scene=json.loads((a.root/'private_ground_truth'/'scene.json').read_text())
    runs=[]
    for p in sorted((a.root/'da3').glob('path_*/room_hypothesis.json')):
        h=json.loads(p.read_text()); name=p.parent.name
        dims=h.get('dimensions'); complete=h.get('status')=='complete_box_hypothesis' and dims and all(x is not None for x in dims)
        room={'room_id':'room-a','shape_dimensions':dims if complete else [None,None,None],
              'wall_height_m':None,'video_wall_length_m':None,
              'walls':[],'openings':[],'complete_surface_pairs':sum(bool(x.get('surface_pair',{}).get('complete')) for x in h.get('axes',[]))}
        if complete:
            room['height_axis_index']=None
        runs.append({'stage':'single_room','house_id':'blender-full-house-17','capture_seed':name,
                     'evaluation_mode':'height_normalized_3m','scale_source':None,'rooms':[room]})
    gt_runs=[]
    for r in runs:
        gt_runs.append({'stage':r['stage'],'house_id':r['house_id'],'capture_seed':r['capture_seed'],
            'rooms':[{'room_id':'room-a','shape_dimensions':[gt_scene['width'],gt_scene['length'],gt_scene['height']],
                      'wall_height_m':gt_scene['height'],'video_wall_length_m':gt_scene['width']+gt_scene['length'],
                      'walls':[],'openings':[]} ]})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    (a.output.with_name(a.output.stem+'-pred.json')).write_text(json.dumps({'schema_version':1,'evaluation_mode':'height_normalized_3m','runs':runs},indent=2))
    (a.output.with_name(a.output.stem+'-gt.json')).write_text(json.dumps({'schema_version':1,'runs':gt_runs},indent=2))
    print(a.output.with_name(a.output.stem+'-pred.json'))
if __name__=='__main__': main()
