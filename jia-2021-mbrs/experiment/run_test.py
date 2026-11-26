# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "torchvision==0.29.2", "kornia==0.9.2", "Pillow==13.0.0", "numpy==2.4.6", "scipy==1.17.1", "scikit-image==0.26.0", "pypdfium2==5.15.2"]
# ///
# Run: .venv/bin/python -B experiment/run_test.py --output experiment/results/run1
"""CPU evaluation of two official MBRS checkpoints and independent attacks."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torchvision import transforms
from common import ROOT, SEED, IMAGES, MODELS, configure, environment, load_model, seed_all, verify_inputs, write_json
from attacks import as_tensor, generate, png_image


def quality(received: torch.Tensor, cover: torch.Tensor):
    a=((received[0].permute(1,2,0)+1)/2).numpy().astype(np.float64)
    b=((cover[0].permute(1,2,0)+1)/2).numpy().astype(np.float64)
    return {'psnr_db':float(peak_signal_noise_ratio(b,a,data_range=1)), 'ssim':float(structural_similarity(b,a,data_range=1,channel_axis=2,win_size=7))}

def csv_write(path: Path, rows) -> None:
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    configure()
    verify_inputs()
    rows=[]
    clean_quality=[]
    examples={}
    settings={'seed':SEED,'device':'cpu','dtype':'float32','threads':2,'interop_threads':1,'eval':True,'deterministic_algorithms':True,'environment':environment(),'jpeg':'Pillow JPEG subsampling=2 (4:2:0)','preprocessing':'Official Resize((int(H*1.1),int(W*1.1))) bilinear PIL, seeded RandomCrop(H,W), RGB ToTensor, Normalize(.5,.5)','serialization':'clamp [-1,1]; scale to [0,255]; round ties to even; uint8 PNG; reopen','decoder_threshold':'>0.5','quality':'PSNR and scikit-image SSIM (7x7, uniform window, sample covariance), RGB, data_range=1; received vs cover','additional_strengths':'GN variance .01, MF3, SP .01 are additional choices: absent from paper Table 2','crop_definition':'official zero-mask on normalized tensor, not white padding stated in PDF; crop sides .19 from results log (~3.5% retained); cropout sqrt(.3) sides; dropout .3 retains encoded samples','resize':'bilinear with antialias, side x.5, restored to model dimensions for fixed-length decoder','noise_output':'Non-JPEG official noises decoded as float32, without clipping or a second serialization; preview PNGs are clipped and rounded','models':{}}
    for model_index,spec in enumerate(MODELS):
        net,_,_,_=load_model(spec)
        settings['models'][spec.name]={'H':spec.side,'W':spec.side,'message_length':spec.length,'with_diffusion':spec.diffusion,'strength_factor':1,'epoch':spec.epoch,'images':{}}
        for image_index,(name,(filename,_)) in enumerate(IMAGES.items()):
            sample_seed=SEED+100*model_index+image_index
            seed_all(sample_seed)
            with Image.open(ROOT/'data'/filename) as opened:
                source=opened.convert('RGB')
            resized=transforms.functional.resize(source,[int(spec.side*1.1)]*2,interpolation=transforms.InterpolationMode.BILINEAR)
            top,left,height,width=transforms.RandomCrop.get_params(resized,[spec.side]*2)
            cover=as_tensor(transforms.functional.crop(resized,top,left,height,width))
            message=torch.from_numpy(np.random.choice([0,1],(1,spec.length)).astype(np.float32))
            folder=args.output/f'{spec.name}-{name}'
            folder.mkdir(exist_ok=True)
            png_image(cover).save(folder/'cover.png')
            with torch.inference_mode():
                encoded=net.encoder(cover,message)
                png_image(encoded).save(folder/'encoded.png')
                original_scores=net.decoder(cover)[0]
                original_bits=(original_scores>.5).int().tolist()
                expected=message[0].int().tolist()
                record={'seed':sample_seed,'message':expected,'resized':list(resized.size),'crop_top_left_hw':[top,left,height,width]}
                settings['models'][spec.name]['images'][name]=record
                common={'model':spec.name,'image':name,'message_length':spec.length}
                rows.append({**common,'attack':'original_cover','strength':'none','matched':int(((original_scores>.5)==message[0].bool()).sum()),'bit_accuracy':sum(a==b for a,b in zip(original_bits,expected,strict=True))/spec.length,'psnr_db':None,'ssim':1.0})
                example={'message':expected,'original_decoded':original_bits,'attacks':{}}
                if spec.diffusion:
                    expanded=net.encoder.message_duplicate_layer(message)
                else:
                    expanded=message
                example['expanded_256']=expanded[0].tolist()
                example['message_map_shape']=[16,16,1]
                for attack_index,(attack,strength,received) in enumerate(generate(folder/'encoded.png',cover)):
                    scores=net.decoder(received)[0]
                    bits=(scores>.5).int().tolist()
                    matched=sum(a==b for a,b in zip(bits,expected,strict=True))
                    metrics=quality(received,cover)
                    rows.append({**common,'attack':attack,'strength':strength,'matched':matched,'bit_accuracy':matched/spec.length,**metrics})
                    suffix=attack+('-'+strength if attack=='jpeg' else '')
                    png_image(received).save(folder/f'{suffix}.png')
                    example['attacks'][suffix]={'decoded':bits,'scores':scores.tolist(),'mismatch_indices':[i for i,(a,b) in enumerate(zip(bits,expected,strict=True)) if a!=b]}
                    if attack=='encoded_png':
                        clean_quality.append({**common,**metrics})
                        cover_pixels=np.array(png_image(cover)).astype(int)
                        encoded_pixels=np.array(png_image(received)).astype(int)
                        y=x=spec.side//2-4
                        example['pixel_block']={'channel':'R','top_left_yx':[y,x],'cover':cover_pixels[y:y+8,x:x+8,0].tolist(),'encoded':encoded_pixels[y:y+8,x:x+8,0].tolist(),'difference':(encoded_pixels-cover_pixels)[y:y+8,x:x+8,0].tolist()}
                    seed_all(sample_seed+1000+attack_index)
            examples[f'{spec.name}-{name}']=example
            print(spec.name,name,'clean',clean_quality[-1],flush=True)
    write_json(args.output/'results.json',rows)
    csv_write(args.output/'measurements.csv',rows)
    write_json(args.output/'quality.json',clean_quality)
    csv_write(args.output/'quality.csv',clean_quality)
    write_json(args.output/'settings.json',settings)
    write_json(args.output/'example_values.json',examples)

if __name__=='__main__':
    main()
