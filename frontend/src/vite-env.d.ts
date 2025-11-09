/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_MAPBOX_TOKEN: string;
  // add more env vars here if you use others
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
