import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

import {loadStyleLibrary, orderedStyles} from './handdrawn-style-library.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const {catalog} = loadStyleLibrary(root);

console.log(`Hand-drawn style library v${catalog.version} — ${catalog.styles.length} styles`);
console.log(`Default: ${catalog.default_style}`);
if (catalog.contact_sheet) console.log(`Contact sheet: ${catalog.contact_sheet}`);
console.log('');
for (const style of orderedStyles(root)) {
  const marker = style.id === catalog.default_style ? ' ★ default' : '';
  console.log(`${String(style.order).padStart(2, '0')}  ${style.id}${marker}`);
  console.log(`    ${style.name_zh} / ${style.name_en}`);
  console.log(`    ${style.summary}`);
  console.log(`    Best for: ${style.best_for.join('、')}`);
  if (style.example_image) console.log(`    Example: ${style.example_image}`);
  console.log('');
}
