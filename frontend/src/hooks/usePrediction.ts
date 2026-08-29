import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, predict } from '../lib/api';
import type { PredictionResponse, RequestStatus } from '../types';

interface UsePredictionResult {
  status: RequestStatus;
  result: PredictionResponse | null;
  error: ApiError | null;
  isLoading: boolean;
  analyze: (file: File) => Promise<void>;
  reset: () => void;
  /** Re-runs the most recent file, for the error card's Retry action. */
  retry: () => Promise<void>;
}

/**
 * Owns the whole request lifecycle: idle -> loading -> success | error.
 *
 * Components read this state and render it; none of them hold request state of
 * their own. An in-flight request is aborted if a new one starts or the
 * component unmounts, so a slow response can never overwrite a newer result.
 */
export function usePrediction(): UsePredictionResult {
  const [status, setStatus] = useState<RequestStatus>('idle');
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const controllerRef = useRef<AbortController | null>(null);
  const lastFileRef = useRef<File | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      controllerRef.current?.abort();
    };
  }, []);

  const analyze = useCallback(async (file: File): Promise<void> => {
    // Supersede any request still in flight.
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    lastFileRef.current = file;

    setStatus('loading');
    setError(null);
    setResult(null);

    try {
      const response = await predict(file, controller.signal);
      if (!mountedRef.current || controller.signal.aborted) return;
      setResult(response);
      setStatus('success');
    } catch (cause: unknown) {
      if (!mountedRef.current || controller.signal.aborted) return;
      setError(
        cause instanceof ApiError
          ? cause
          : new ApiError('INTERNAL_ERROR', 'An unexpected error occurred.'),
      );
      setStatus('error');
    }
  }, []);

  const retry = useCallback(async (): Promise<void> => {
    const file = lastFileRef.current;
    if (file) await analyze(file);
  }, [analyze]);

  const reset = useCallback((): void => {
    controllerRef.current?.abort();
    lastFileRef.current = null;
    setStatus('idle');
    setResult(null);
    setError(null);
  }, []);

  return {
    status,
    result,
    error,
    isLoading: status === 'loading',
    analyze,
    reset,
    retry,
  };
}
