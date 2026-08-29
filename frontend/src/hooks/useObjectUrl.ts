import { useEffect, useState } from 'react';

/**
 * An object URL for `file`, revoked whenever the file changes or the component
 * unmounts. Without the revoke, every selected image leaks for the page's life.
 */
export function useObjectUrl(file: File | null): string | null {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  return url;
}
