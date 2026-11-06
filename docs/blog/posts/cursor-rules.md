---
authors:
  - jxnl
categories:
  - Contributing
comments: true
date: 2025-03-18
description:
  Learn how Instructor's Cursor rules improve Git workflows for contributors, making AI-assisted coding more organized.
draft: false
slug: cursor-rules-for-better-git-practices
tags:
  - Git
  - Cursor
  - Contributing
  - Best Practices
---

# Instructor Adopting Cursor Rules

AI-assisted coding is changing how we use version control. Many developers now use what I call "vibe coding" - coding with AI help. This creates new challenges with Git. Today I'll share how we're using Cursor rules in Instructor to solve these problems.

<!-- more -->

## The Git Problem When Coding with AI

In my blog post [Version Control for the Vibe Coder (Part 1)](https://jxnl.co/writing/2025/03/18/version-control-for-the-vibe-coder-part-1/), I wrote about the problem:

> "Imagine this: you open Cursor, ask it to build a feature in YOLO-mode, and let it rip. You feel great as you watch code materialize... until you realize you haven't made a single commit, your branch is a mess, and you have no idea how to organize these changes for review."

This happens often. When using AI tools like Cursor, we focus on creating code quickly but forget about version control. This leads to big, messy commits that are hard to review.

## How Cursor Rules Help

We've added Cursor rules to Instructor. These rules help standardize Git workflows inside Cursor. The rules are simple markdown files in the `.cursor/rules` directory that guide Cursor when working with your code.

As I wrote in [Version Control for the Vibe Coder (Part 2)](https://jxnl.co/writing/2025/03/18/version-control-for-the-vibe-coder-part-2/):

> "Add rules to `.cursor/rules` to instruct Cursor clearly and repeatedly... The real key to success with Git is much simpler: Make Small, Frequent Commits... Let Cursor Handle the Rest."

This balances fast AI coding with good teamwork practices.

## How Our Cursor Rules Help Contributors

If you want to contribute to Instructor, our Cursor rules will make it easier. Here's how:

### 1. Better Branching and Commits

The rules help Cursor suggest good Git practices. When building a new feature, Cursor will help you:

- Create well-named branches
- Make small commits with clear messages
- Format PR descriptions correctly

### 2. Simpler PR Process

Our rules define how to create and manage pull requests:

- Format PR descriptions
- Add the right reviewers
- Use stacked PRs for big features (as I explain in my Part 2 blog post)

### 3. Keeping Docs Updated

The rules remind you to update docs when code changes, which keeps our project docs accurate.

## Getting Started

If you're new to Instructor or Cursor, here's how to use these rules:

1. **Install Cursor**: Download it from [cursor.sh](https://cursor.sh/)
2. **Clone Instructor**: `git clone https://github.com/instructor-ai/instructor.git`
3. **Open in Cursor**: The `.cursor/rules` will load automatically
4. **Make changes**: Let Cursor guide your Git workflow
5. **Create a PR**: Follow Cursor's suggestions

You don't need to remember all the Git commands. The rules will help Cursor suggest the right steps.

## Stacked PRs for Bigger Features

One key practice in our rules is stacked PRs. As I explain:

> "Stacked pull requests are a powerful workflow for building complex features incrementally. Instead of one massive PR, you create a series of smaller, dependent PRs that build upon each other."

This helps Instructor because it allows:

- Focused code reviews
- Easier merging of changes
- Better organization of big features
- Clear documentation of decisions

The rules show you how to make and manage stacked PRs without confusion.

## Keeping the Human Touch

A big benefit of Cursor rules is keeping people central to the process. While AI helps write code, the rules ensure:

- Code changes stay clear and reviewable
- Docs stay current
- Commit history tells a clear story
- Contributors get credit for their work

## Try It Out

I invite you to make a PR to Instructor with small changes. Using AI-assisted coding with Git through Cursor rules makes contributing easier and more fun.

Start small - fix a typo or add an example to the cookbook. Open the repo in Cursor and let the rules guide you through making a clean PR. This lets you focus on writing good code instead of figuring out Git commands.

Remember: "The most important Git skill is making regular, small commits. Everything else - bisecting, stacked PRs, complex rebases - these are just tools that Cursor can handle for you."

With Cursor rules, you get fast AI coding plus good team practices.

If you want to add Cursor rules to your own open source projects, I can help! Reach out to me on Twitter at [@jxnlco](https://twitter.com/jxnlco) and I'll share what we've learned.

Happy coding!