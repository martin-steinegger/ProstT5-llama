
import { viteSingleFile } from 'vite-plugin-singlefile';
import path from 'path';
import fs from 'fs';
import zlib from 'zlib';

const MAX_BUNDLE_SIZE = 1.5 * 1024 * 1024; // only increase when absolutely necessary

const GUIDE_FOR_FRONTEND = `
<!--
  This is a single file build of the frontend.
  It is automatically generated by the build process.
  Do not edit this file directly.
  To make changes, refer to the "Web UI" section in the README.
-->
`.trim();

const BUILD_PLUGINS = [
  viteSingleFile(),
  (function llamaCppPlugin() {
    let config;
    return {
      name: 'llamacpp:build',
      apply: 'build',
      async configResolved(_config) {
        config = _config;
      },
      writeBundle() {
        const outputIndexHtml = path.join(config.build.outDir, 'index.html');
        const content = GUIDE_FOR_FRONTEND + '\n' + fs.readFileSync(outputIndexHtml, 'utf-8');
        const compressed = zlib.gzipSync(Buffer.from(content, 'utf-8'), { level: 9 });

        // because gzip header contains machine-specific info, we must remove these data from the header
        // timestamp
        compressed[0x4] = 0;
        compressed[0x5] = 0;
        compressed[0x6] = 0;
        compressed[0x7] = 0;
        // OS
        compressed[0x9] = 0;

        if (compressed.byteLength > MAX_BUNDLE_SIZE) {
          throw new Error(
            `Bundle size is too large (${Math.ceil(compressed.byteLength / 1024)} KB).\n` +
            `Please reduce the size of the frontend or increase MAX_BUNDLE_SIZE in vite.config.js.\n`,
          );
        }

        const targetOutputFile = path.join(config.build.outDir, '../../public/index.html.gz');
        fs.writeFileSync(targetOutputFile, compressed);
      }
    }
  })(),
];

/** @type {import('vite').UserConfig} */
export default {
  plugins: process.env.ANALYZE ? [] : BUILD_PLUGINS,
};