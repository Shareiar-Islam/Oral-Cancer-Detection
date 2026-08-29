/** Mirrors the backend's Pydantic response models exactly. */

export interface ImageInfo {
  /** [width, height] as uploaded. */
  original_size: [number, number];
  /** [width, height] the model actually saw. */
  processed_size: [number, number];
  exif_corrected: boolean;
  format: string;
  mode: string;
}

export interface PredictionResponse {
  prediction: string;
  /** P(Cancer), independent of which label won. */
  probability: number;
  /** Probability of the predicted class. */
  confidence: number;
  threshold: number;
  /** Pre-activation model output. */
  raw_output: number;
  inference_time_ms: number;
  image: ImageInfo;
}

/**
 * Returned by GET /api/health. The app no longer polls this — it is kept as
 * the documented shape of the endpoint for anyone calling the API directly.
 */
export interface HealthResponse {
  status: 'ok' | 'degraded';
  model_loaded: boolean;
  device: string;
}

/** Returned by GET /api/model-info. Useful for auditing a deployment. */
export interface ModelInfoResponse {
  architecture: string;
  checkpoint: string;
  device: string;
  input_size: [number, number];
  num_output_units: number;
  activation: string;
  outputs_probability: boolean;
  threshold: number;
  positive_class_index: number;
  class_names: Record<string, string>;
  normalization: { mean: number[]; std: number[] };
  resize_mode: string;
  accepted_mime_types: string[];
  max_upload_mb: number;
  max_batch_files: number;
  load_path: string;
  missing_keys: number;
  unexpected_keys: number;
  checkpoint_metadata: Record<string, unknown>;
}

/** The envelope every backend failure uses. */
export interface ApiErrorBody {
  error: { code: string; message: string };
}

/** Error codes the UI reacts to specifically. */
export type ApiErrorCode =
  | 'MODEL_NOT_LOADED'
  | 'MODEL_LOAD_FAILED'
  | 'UNSUPPORTED_MEDIA_TYPE'
  | 'INVALID_IMAGE'
  | 'FILE_TOO_LARGE'
  | 'EMPTY_UPLOAD'
  | 'TOO_MANY_FILES'
  | 'INFERENCE_FAILED'
  | 'VALIDATION_ERROR'
  | 'NETWORK_ERROR'
  | 'TIMEOUT'
  | 'CLIENT_VALIDATION'
  | 'INTERNAL_ERROR';

export type RequestStatus = 'idle' | 'loading' | 'success' | 'error';
