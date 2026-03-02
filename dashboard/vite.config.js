import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// IMPORTANT for GitHub Pages:
// base must be "/<repo-name>/" for project pages.
// We set it automatically from env var in GitHub Actions, else default "/".
export default defineConfig(({ mode }) => {
  const repo = process.env.GITHUB_REPOSITORY?.split("/")[1];
  const base = process.env.GITHUB_PAGES === "true" && repo ? `/${repo}/` : "/";
  return {
    plugins: [react()],
    base,
    server: {
      port: 5173,
      strictPort: true
    }
  };
});
