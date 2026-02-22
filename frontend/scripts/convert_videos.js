#!/usr/bin/env node
// Batch-convert videos in frontend/public/videos to MP4 (H.264 + AAC)
// Usage: node scripts/convert_videos.js
// Requires: ffmpeg available in PATH

import * as fs from 'fs';
import * as path from 'path';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const VIDEOS_DIR = path.resolve(__dirname, '..', 'public', 'videos');
const OUT_DIR = path.join(VIDEOS_DIR, 'converted');

if (!fs.existsSync(VIDEOS_DIR)) {
  console.error('Videos directory not found:', VIDEOS_DIR);
  process.exit(1);
}
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

const exts = ['.mp4', '.webm', '.ogg', '.mov', '.mkv'];
const files = fs.readdirSync(VIDEOS_DIR).filter(f => {
  const lower = f.toLowerCase();
  return exts.some(e => lower.endsWith(e)) && f !== 'index.json';
});

if (files.length === 0) {
  console.log('No video files found in', VIDEOS_DIR);
  process.exit(0);
}

console.log('Found', files.length, 'files. Converting to', OUT_DIR);

files.forEach(file => {
  const inPath = path.join(VIDEOS_DIR, file);
  // produce safe output name
  const base = path.parse(file).name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_\-]/g, '');
  const outName = base + '.mp4';
  const outPath = path.join(OUT_DIR, outName);

  // If input already mp4 and reasonably encoded, copy it instead of re-encode
  const isMp4 = file.toLowerCase().endsWith('.mp4');
  if (isMp4) {
    console.log(`Copying ${file} -> converted/${outName}`);
    try { fs.copyFileSync(inPath, outPath); } catch (err) { console.error('Copy failed', err); }
    return;
  }

  // ffmpeg args: -y overwrite, input, libx264, preset fast, crf 23, aac audio
  const args = ['-y', '-i', inPath, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', '-b:a', '128k', outPath];
  console.log('Converting', file);
  const res = spawnSync('ffmpeg', args, { stdio: 'inherit' });
  if (res.error) {
    console.error('ffmpeg failed for', file, res.error);
  } else if (res.status !== 0) {
    console.error('ffmpeg exited with code', res.status, 'for', file);
  } else {
    console.log('Converted ->', outName);
  }
});

console.log('Done. Converted files are in', OUT_DIR);
