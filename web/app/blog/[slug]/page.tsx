import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { MDXRemote } from "next-mdx-remote/rsc";
import { getBlogPost, getBlogSlugs } from "@/lib/blog";

type Props = {
  params: Promise<{ slug: string }>;
};

export function generateStaticParams() {
  return getBlogSlugs().map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  try {
    const post = getBlogPost(slug);
    return {
      title: post.title,
      description: post.excerpt
    };
  } catch {
    return {};
  }
}

export default async function BlogPostPage({ params }: Props) {
  const { slug } = await params;
  let post;

  try {
    post = getBlogPost(slug);
  } catch {
    notFound();
  }

  return (
    <main className="narrow py-16">
      <p className="masthead mb-4">
        {new Date(post.date).toLocaleDateString("en-IN", { dateStyle: "long" })} · {post.readTime}
      </p>
      <h1 className="section-title">{post.title}</h1>
      <p className="serif-italic mt-6 text-2xl leading-snug text-muted">{post.excerpt}</p>
      <article className="prose prose-invert prose-editorial mt-12 max-w-none prose-headings:tracking-normal prose-p:text-lg prose-p:leading-8">
        <MDXRemote source={post.content} />
      </article>
    </main>
  );
}
