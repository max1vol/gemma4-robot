import fs from 'node:fs';
import { Readable } from 'node:stream';
import { resolveRepoPath } from '$lib/dataset.js';

export function GET({ url }) {
  const relativePath = url.searchParams.get('path');
  const absolutePath = resolveRepoPath(relativePath);
  if (!absolutePath || !fs.existsSync(absolutePath)) {
    return new Response('Audio not found', { status: 404 });
  }

  const stat = fs.statSync(absolutePath);
  const stream = fs.createReadStream(absolutePath);
  return new Response(Readable.toWeb(stream), {
    headers: {
      'content-type': 'audio/wav',
      'content-length': String(stat.size),
      'accept-ranges': 'bytes',
      'cache-control': 'private, max-age=60'
    }
  });
}
