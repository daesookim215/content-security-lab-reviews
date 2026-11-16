import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import crypto from 'node:crypto';
const root=path.dirname(fileURLToPath(import.meta.url));
const sources=[
  ['https://codeload.github.com/ando-khachatryan/HiDDeN/zip/556f76dd0602e4351ed4c19bd2ee87d50ef3c0de','ando-pinned.zip','13ded3eb62de8ac996c90aa161a7a0856cdd9f2497b5953439f079b4eb753ca2'],
  ['https://sipi.usc.edu/database/misc/4.2.05.tiff','4.2.05.tiff','ae9eba12b80df955e47585bf4a16332ea0dadc753668d3457c509564934dbd40'],
  ['https://sipi.usc.edu/database/misc/boat.512.tiff','boat.512.tiff','6bc7319b1e3b44ce0b2a571ac39a28bac94437d812057d1afb2cf78ea3a9a35f'],
];
export async function run(){
  const cache=path.join(root,'.cache');
  await fs.mkdir(cache,{recursive:true});
  for(const [url,name,expected] of sources){
    const response=await fetch(url);
    if(!response.ok)throw new Error(name+': '+response.status);
    const bytes=Buffer.from(await response.arrayBuffer());
    const sha256=crypto.createHash('sha256').update(bytes).digest('hex');
    if(sha256!==expected)throw new Error(name+': SHA-256 mismatch');
    await fs.writeFile(path.join(cache,name),bytes);
    console.log({name,status:response.status,bytes:bytes.length,sha256});
  }
}
