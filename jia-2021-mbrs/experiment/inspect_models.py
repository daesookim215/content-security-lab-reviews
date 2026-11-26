# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "torchvision==0.29.2", "kornia==0.9.2"]
# ///
# Run: .venv/bin/python -B experiment/inspect_models.py
"""Record strict loads, module trees, parameters and measured layer shapes."""
from pathlib import Path
import torch
from common import MODELS, configure, load_model, verify_inputs, write_json


def main() -> None:
    configure()
    verify_inputs()
    result = {}
    trees = []
    for spec in MODELS:
        net,dis,ec_state,d_state = load_model(spec)
        entry = {'settings':{'H':spec.side,'W':spec.side,'message_length':spec.length,'with_diffusion':spec.diffusion,'strength_factor':1,'epoch':spec.epoch},'strict_load':{'EC':'all keys matched','D':'all keys matched'},'checkpoints':{'EC':{k:list(v.shape) for k,v in ec_state.items()},'D':{k:list(v.shape) for k,v in d_state.items()}},'networks':{}}
        image=torch.zeros(1,3,spec.side,spec.side)
        message=torch.zeros(1,spec.length)
        for name,module,args in [('encoder',net.encoder,(image,message)),('decoder',net.decoder,(image,)),('discriminator',dis,(image,))]:
            shapes={}
            def hook(path):
                def record(layer,inputs,output):
                    shapes[path]={'type':type(layer).__name__,'inputs':[list(t.shape) for t in inputs],'output':list(output.shape),'parameters_recursive':sum(p.numel() for p in layer.parameters()),'parameters_direct':sum(p.numel() for p in layer.parameters(recurse=False))}
                return record
            handles=[layer.register_forward_hook(hook(path)) for path,layer in module.named_modules()]
            with torch.inference_mode():
                module(*args)
            for handle in handles:
                handle.remove()
            entry['networks'][name]={'parameters':sum(p.numel() for p in module.parameters()),'layers':shapes}
            trees.append(f'{spec.name} / {name}\n{module}\n')
        result[spec.name]=entry
        print(spec.name,{k:v['parameters'] for k,v in entry['networks'].items()},flush=True)
    write_json(Path('experiment/shapes_params.json'),result)
    Path('experiment/module_tree.txt').write_text('\n'.join(trees))

if __name__=='__main__':
    main()
