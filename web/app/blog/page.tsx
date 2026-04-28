import type { Metadata } from "next";
import Link from "next/link";
import { getAllBlogPosts } from "@/lib/blog";

export const metadata: Metadata = {
  title: "Blog",
  description: "Sanketra essays on Hindi input, local speech-to-text, and PC control."
};

export default function BlogPage() {
  const posts = getAllBlogPosts();

  return (
    <main className="wrap py-16">
      <p className="masthead mb-4">Blog</p>
      <h1 className="section-title">Sharp notes from the build.</h1>
      <div className="mt-12 grid gap-5">
        {posts.map((post) => (
          <Link key={post.slug} href={`/blog/${post.slug}`} className="card block p-6 transition hover:border-saffron">
            <div className="flex flex-wrap gap-4 font-mono text-[0.72rem] uppercase tracking-[0.14em] text-faint">
              <span>{new Date(post.date).toLocaleDateString("en-IN", { dateStyle: "medium" })}</span>
              <span>{post.readTime}</span>
            </div>
            <h2 className="mt-4 text-3xl font-semibold tracking-normal">{post.title}</h2>
            <p className="mt-3 max-w-3xl text-muted">{post.excerpt}</p>
          </Link>
        ))}
      </div>
    </main>
  );
}
