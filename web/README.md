# Sanketra Web

Public marketing website for `sanketra.app`, built with Next.js 15 App Router.

## Local build

```bash
npm install
npm run build
```

## Development

```bash
npm run dev
```

## Deploy

Set the Vercel project root to `web/`, keep the build command as `npm run build`,
and point `sanketra.app` DNS at the Vercel project.

## Content

- Blog posts live in `content/blog/*.mdx`.
- `/changelog` reads `../CHANGELOG.md` at build time and revalidates every hour.
- Replace `public/og-image.png`, add `public/demo.mp4`, and add GIFs under
  `public/gifs/` before launch.
