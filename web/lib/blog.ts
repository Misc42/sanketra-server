import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";

const blogDir = path.join(process.cwd(), "content/blog");

export type BlogPost = {
  slug: string;
  title: string;
  date: string;
  excerpt: string;
  readTime: string;
  content: string;
};

function estimateReadTime(content: string) {
  const words = content.trim().split(/\s+/).filter(Boolean).length;
  return `${Math.max(1, Math.ceil(words / 220))} min read`;
}

export function getBlogSlugs() {
  return fs
    .readdirSync(blogDir)
    .filter((file) => file.endsWith(".mdx"))
    .map((file) => file.replace(/\.mdx$/, ""));
}

export function getBlogPost(slug: string): BlogPost {
  const file = path.join(blogDir, `${slug}.mdx`);
  const raw = fs.readFileSync(file, "utf8");
  const { data, content } = matter(raw);

  return {
    slug: String(data.slug ?? slug),
    title: String(data.title),
    date: String(data.date),
    excerpt: String(data.excerpt),
    readTime: estimateReadTime(content),
    content
  };
}

export function getAllBlogPosts() {
  return getBlogSlugs()
    .map(getBlogPost)
    .sort((a, b) => Date.parse(b.date) - Date.parse(a.date));
}
