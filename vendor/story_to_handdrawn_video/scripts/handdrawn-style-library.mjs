import {existsSync, readFileSync} from 'node:fs';
import {resolve} from 'node:path';

export const styleLibraryPath = (root) =>
  resolve(root, 'references/handdrawn-style-library.json');

const normalizeSelector = (value) => String(value || '').trim().toLocaleLowerCase();

export const loadStyleLibrary = (root) => {
  const path = styleLibraryPath(root);
  if (!existsSync(path)) throw new Error(`Missing hand-drawn style library: ${path}`);

  const catalog = JSON.parse(readFileSync(path, 'utf8'));
  if (!Array.isArray(catalog.styles) || catalog.styles.length === 0) {
    throw new Error(`Hand-drawn style library has no styles: ${path}`);
  }

  const selectors = new Map();
  for (const style of catalog.styles) {
    if (!style.id || !style.name_zh || !Number.isInteger(style.order)) {
      throw new Error(`Invalid style record in ${path}`);
    }
    const keys = [style.id, style.name_zh, style.name_en, String(style.order), ...(style.aliases || [])];
    for (const key of keys) {
      const normalized = normalizeSelector(key);
      if (!normalized) continue;
      const prior = selectors.get(normalized);
      if (prior && prior !== style.id) {
        throw new Error(`Duplicate style selector "${key}" for ${prior} and ${style.id}`);
      }
      selectors.set(normalized, style.id);
    }

    if (!style.profile_file && !Array.isArray(style.prompt_blocks)) {
      throw new Error(`Style ${style.id} needs profile_file or prompt_blocks`);
    }
    if (style.example_image && !existsSync(resolve(root, style.example_image))) {
      throw new Error(`Style ${style.id} is missing example image: ${resolve(root, style.example_image)}`);
    }
  }

  if (!catalog.styles.some((style) => style.id === catalog.default_style)) {
    throw new Error(`Default style ${catalog.default_style} is not defined in ${path}`);
  }
  if (catalog.contact_sheet && !existsSync(resolve(root, catalog.contact_sheet))) {
    throw new Error(`Style library is missing contact sheet: ${resolve(root, catalog.contact_sheet)}`);
  }

  return {catalog, path, selectors};
};

export const resolveStyle = (root, selector) => {
  const library = loadStyleLibrary(root);
  const requested = normalizeSelector(selector || library.catalog.default_style);
  const styleId = library.selectors.get(requested);
  if (!styleId) {
    const available = library.catalog.styles
      .sort((a, b) => a.order - b.order)
      .map((style) => `${style.order}:${style.id}`)
      .join(', ');
    throw new Error(`Unknown --style "${selector}". Available styles: ${available}`);
  }

  const style = library.catalog.styles.find((item) => item.id === styleId);
  const profilePath = style.profile_file ? resolve(root, style.profile_file) : null;
  if (profilePath && !existsSync(profilePath)) {
    throw new Error(`Style ${style.id} is missing profile file: ${profilePath}`);
  }
  const prompt = profilePath
    ? readFileSync(profilePath, 'utf8').trim()
    : style.prompt_blocks.join('\n');
  const references = (style.reference_images || []).map((reference) => ({
    ...reference,
    absolute_path: resolve(root, reference.path),
  }));
  const missingReferences = references
    .filter((reference) => !existsSync(reference.absolute_path))
    .map((reference) => reference.absolute_path);
  if (missingReferences.length > 0) {
    throw new Error(`Style ${style.id} is missing reference images:\n${missingReferences.join('\n')}`);
  }

  return {
    ...style,
    prompt,
    profile_path: profilePath,
    references,
    example_path: style.example_image ? resolve(root, style.example_image) : null,
    contact_sheet_path: library.catalog.contact_sheet
      ? resolve(root, library.catalog.contact_sheet)
      : null,
    library_path: library.path,
    library_version: library.catalog.version,
    is_default: style.id === library.catalog.default_style,
  };
};

export const orderedStyles = (root) =>
  [...loadStyleLibrary(root).catalog.styles].sort((a, b) => a.order - b.order);
