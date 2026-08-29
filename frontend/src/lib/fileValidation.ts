import { ApiError } from './api';

/** Mirrors the backend's accepted list. The server re-validates by decoding. */
export const ACCEPTED_MIME_TYPES: readonly string[] = [
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/bmp',
  'image/tiff',
];

export const ACCEPTED_EXTENSIONS: readonly string[] = [
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
  '.bmp',
  '.tif',
  '.tiff',
];

/** Must match the backend's MAX_UPLOAD_MB. */
export const MAX_UPLOAD_MB = 10;
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Cheap client-side pre-check so obvious mistakes never cost a round trip.
 * This is a convenience, NOT a security boundary — the server validates by
 * actually decoding the bytes.
 */
export function validateFile(file: File): ApiError | null {
  if (file.size === 0) {
    return new ApiError('EMPTY_UPLOAD', `"${file.name}" is empty. Please choose another image.`);
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    return new ApiError(
      'FILE_TOO_LARGE',
      `"${file.name}" is ${formatBytes(file.size)}, over the ${MAX_UPLOAD_MB} MB limit. Try a smaller or more compressed image.`,
    );
  }

  // Some browsers report an empty type for less common formats, so fall back
  // to the extension rather than rejecting a file the server would accept.
  const typeOk = ACCEPTED_MIME_TYPES.includes(file.type);
  const extensionOk = ACCEPTED_EXTENSIONS.some((ext) =>
    file.name.toLowerCase().endsWith(ext),
  );

  if (!typeOk && !extensionOk) {
    return new ApiError(
      'CLIENT_VALIDATION',
      `"${file.name}" does not look like a supported image. Use JPEG, PNG, WebP, BMP, or TIFF.`,
    );
  }

  return null;
}
